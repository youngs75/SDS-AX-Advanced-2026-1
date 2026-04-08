"""CLI를 위한 LangGraph 서버 수명주기 관리.

`langgraph dev` 서버 프로세스 시작/중지 및 필수 `langgraph.json` 구성 파일 생성을 처리합니다.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess  # noqa: S404
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 2024
_HEALTH_POLL_INTERVAL_LOCAL = 0.1
_HEALTH_POLL_INTERVAL_REMOTE = 0.3
_HEALTH_TIMEOUT = 60
_SHUTDOWN_TIMEOUT = 5


def _port_in_use(host: str, port: int) -> bool:
    """포트가 이미 사용 중인지 확인하세요.

    Args:
        host: 확인해야 할 호스트입니다.
        port: 확인할 포트입니다.

    Returns:


    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True
        else:
            return False


def _find_free_port(host: str) -> int:
    """해당 호스트에서 사용 가능한 포트를 찾습니다.

    Args:
        host: 바인딩할 호스트입니다.

    Returns:
        사용 가능한 포트 번호입니다.

    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def get_server_url(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> str:
    """서버 기본 URL을 구축합니다.

    Args:
        host: 서버 호스트.
        port: 서버 포트.

    Returns:
        기본 URL 문자열.

    """
    return f"http://{host}:{port}"


def generate_langgraph_json(
    output_dir: str | Path,
    *,
    graph_ref: str = "./server_graph.py:graph",
    env_file: str | None = None,
    checkpointer_path: str | None = None,
) -> Path:
    """`langgraph dev`에 대한 `langgraph.json` 구성 파일을 생성합니다.

    Args:
        output_dir: 구성 파일을 쓸 디렉터리입니다.
        graph_ref: Python 모듈:그래프에 대한 변수 참조.
        env_file: env 파일의 선택적 경로입니다.
        checkpointer_path: `BaseCheckpointSaver`을 생성하는 비동기 컨텍스트 관리자에 대한 경로를 가져옵니다. 설정되면
                           서버는 체크포인트 데이터를 메모리 내 대신 디스크에 유지합니다.

    Returns:
        생성된 구성 파일의 경로입니다.

    """
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {
            "agent": graph_ref,
        },
    }
    if env_file:
        config["env"] = env_file
    if checkpointer_path:
        config["checkpointer"] = {"path": checkpointer_path}

    output_path = Path(output_dir) / "langgraph.json"
    output_path.write_text(json.dumps(config, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Scoped env-var management
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _scoped_env_overrides(
    overrides: dict[str, str],
) -> Iterator[None]:
    """env-var 재정의를 적용하고 예외가 발생한 경우에만 롤백합니다.

    일시적인 `os.environ` 변형에 대한 우려를 하위 프로세스 관리에서 분리하여 둘 다 독립적으로 테스트할 수 있도록 합니다.

    일반 종료 시 재정의는 그대로 유지됩니다(호출자가 이를 "유지"합니다). 예외적으로 이전 값이 복원되므로 다음 시도는 알려진 양호한 상태에서
    시작됩니다.

    Args:
        overrides: `os.environ`에 설정할 키/값 쌍입니다.

    Yields:
        발신자에게 제어합니다.

    """
    prev: dict[str, str | None] = {}
    for key, val in overrides.items():
        prev[key] = os.environ.get(key)
        os.environ[key] = val
    try:
        yield
    except Exception:
        for key, old_val in prev.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        raise


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------


async def wait_for_server_healthy(
    url: str,
    *,
    timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    process: subprocess.Popen | None = None,
    read_log: Callable[[], str] | None = None,
    local: bool = False,
) -> None:
    """응답할 때까지 LangGraph 서버 상태 엔드포인트를 폴링합니다.

    Args:
        url: 서버 기본 URL(상태 끝점은 `{url}/ok`)입니다.
        timeout: 최대 대기 시간(초)입니다.
        process: 선택적 하위 프로세스 핸들. 프로세스가 일찍 종료되면 시간 초과를 기다리는 대신 빠르게 실패합니다.
        read_log: 로그 파일 내용을 반환하는 선택적 호출 가능(초기 종료 시 오류 메시지용).
        local: 로컬 서버에는 더 짧은 폴링 간격을 사용하십시오.

    Raises:
        RuntimeError: 서버가 시간 내에 정상화되지 않는 경우.

    """
    import httpx

    poll_interval = (
        _HEALTH_POLL_INTERVAL_LOCAL if local else _HEALTH_POLL_INTERVAL_REMOTE
    )
    health_url = f"{url}/ok"
    deadline = time.monotonic() + timeout
    last_status: int | None = None
    last_exc: Exception | None = None

    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                output = read_log() if read_log else ""
                msg = f"Server process exited with code {process.returncode}"
                if output:
                    msg += f"\n{output[-3000:]}"
                raise RuntimeError(msg)

            try:
                resp = await client.get(health_url, timeout=2)
                if resp.status_code == 200:  # noqa: PLR2004
                    logger.info("Server is healthy at %s", url)
                    return
                last_status = resp.status_code
                logger.debug("Health check returned status %d", resp.status_code)
            except (httpx.TransportError, OSError) as exc:
                logger.debug("Health check attempt failed: %s", exc)
                last_exc = exc

            await asyncio.sleep(poll_interval)

    msg = f"Server did not become healthy within {timeout}s"
    if last_status is not None:
        msg += f" (last status: {last_status})"
    elif last_exc is not None:
        msg += f" (last error: {last_exc})"
    raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Server command / env construction
# ---------------------------------------------------------------------------


def _build_server_cmd(config_path: Path, *, host: str, port: int) -> list[str]:
    """`langgraph dev` 명령줄을 작성하세요.

    Args:
        config_path: `langgraph.json` 구성 파일의 경로입니다.
        host: 바인딩할 호스트입니다.
        port: 바인딩할 포트입니다.

    Returns:
        명령 argv 목록.

    """
    return [
        sys.executable,
        "-m",
        "langgraph_cli",
        "dev",
        "--host",
        host,
        "--port",
        str(port),
        "--no-browser",
        "--no-reload",
        "--config",
        str(config_path),
    ]


def _build_server_env() -> dict[str, str]:
    """서버 하위 프로세스에 대한 환경 사전을 빌드합니다.

    `os.environ`을 복사하고, 필수 플래그를 설정하고, 로컬 개발 서버에 필요하지 않은(방해할 수 있는) 인증 관련 변수를 제거합니다.

    Returns:
        `subprocess.Popen`에 대한 환경 사전입니다.

    """
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LANGGRAPH_AUTH_TYPE"] = "noop"
    for key in (
        "LANGGRAPH_AUTH",
        "LANGGRAPH_CLOUD_LICENSE_KEY",
        "LANGSMITH_CONTROL_PLANE_API_KEY",
        "LANGSMITH_TENANT_ID",
    ):
        env.pop(key, None)
    return env


# ---------------------------------------------------------------------------
# ServerProcess
# ---------------------------------------------------------------------------


class ServerProcess:
    """`langgraph dev` 서버 하위 프로세스를 관리합니다.

    하위 프로세스 수명 주기(시작, 중지, 다시 시작) 및 상태 확인에 중점을 둡니다. 다시 시작을 위한 Env-var 관리(예: 전체 다시 시작이 필요한
    구성 변경)는 `_scoped_env_overrides`에서 처리되므로 이 클래스는 프로세스 관리에 중점을 둡니다.

    """

    def __init__(
        self,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        config_dir: str | Path | None = None,
        owns_config_dir: bool = False,
    ) -> None:
        """서버 프로세스 관리자를 초기화합니다.

        Args:
            host: 서버를 바인딩할 호스트입니다.
            port: 서버를 바인딩할 초기 포트입니다.

                포트가 이미 사용 중인 경우 `start()`에 의해 자동으로 재할당될 수 있습니다.
            config_dir: `langgraph.json`을(를) 포함하는 디렉터리입니다.
            owns_config_dir: `True`일 때 서버는 `stop()`에서 `config_dir`을(를) 삭제합니다.

        """
        self.host = host
        self.port = port
        self.config_dir = Path(config_dir) if config_dir else None
        self._owns_config_dir = owns_config_dir
        self._process: subprocess.Popen | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._log_file: tempfile.NamedTemporaryFile | None = None  # type: ignore[type-arg]
        self._env_overrides: dict[str, str] = {}

    @property
    def url(self) -> str:
        """서버 기본 URL."""
        return get_server_url(self.host, self.port)

    @property
    def running(self) -> bool:
        """서버 프로세스가 실행 중인지 여부입니다."""
        return self._process is not None and self._process.poll() is None

    def _read_log_file(self) -> str:
        """서버 로그 파일 내용을 읽습니다.

        Returns:
            파일 내용을 문자열로 기록합니다(비어 있을 수 있음).

        """
        if self._log_file is None:
            return ""
        try:
            self._log_file.flush()
            return Path(self._log_file.name).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            logger.warning(
                "Failed to read server log file %s",
                self._log_file.name,
                exc_info=True,
            )
            return ""

    async def start(
        self,
        *,
        timeout: float = _HEALTH_TIMEOUT,  # noqa: ASYNC109
    ) -> None:
        """`langgraph dev` 서버를 시작하고 정상 상태가 될 때까지 기다립니다.

        Args:
            timeout: 서버가 정상 상태가 될 때까지 기다리는 최대 시간(초)입니다.

        Raises:
            RuntimeError: 서버가 시작되지 않거나 정상 상태가 되는 경우.

        """
        if self.running:
            return

        work_dir = self.config_dir
        if work_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="deepagents_server_")
            work_dir = Path(self._temp_dir.name)

        config_path = work_dir / "langgraph.json"
        if not config_path.exists():
            msg = (
                f"langgraph.json not found in {work_dir}. "
                "Call generate_langgraph_json() first."
            )
            raise RuntimeError(msg)

        if _port_in_use(self.host, self.port):
            self.port = _find_free_port(self.host)
            logger.info("Default port in use, using port %d instead", self.port)

        cmd = _build_server_cmd(config_path, host=self.host, port=self.port)
        env = _build_server_env()

        logger.info("Starting langgraph dev server: %s", " ".join(cmd))
        self._log_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            prefix="deepagents_server_log_",
            suffix=".txt",
            delete=False,
            mode="w",
            encoding="utf-8",
        )
        self._process = subprocess.Popen(  # noqa: S603, ASYNC220
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

        try:
            await wait_for_server_healthy(
                self.url,
                timeout=timeout,
                process=self._process,
                read_log=self._read_log_file,
                local=True,
            )
        except Exception:
            self.stop()
            raise

    def _stop_process(self) -> None:
        """서버 하위 프로세스와 해당 로그 파일만 중지하십시오.

        `stop()`과 달리 이는 구성 디렉터리나 임시 디렉터리를 정리하지 않으므로 동일한 구성으로 서버를 다시 시작할 수 있습니다.

        """
        if self._process is None:
            return

        if self._process.poll() is None:
            logger.info("Stopping langgraph dev server (pid=%d)", self._process.pid)
            try:
                self._process.send_signal(signal.SIGTERM)
                self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, killing")
                self._process.kill()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Server process pid=%d did not exit after SIGKILL",
                        self._process.pid,
                    )
            except OSError:
                logger.warning("Error stopping server", exc_info=True)

        self._process = None

        if self._log_file is not None:
            try:
                self._log_file.close()
                Path(self._log_file.name).unlink()
            except OSError:
                logger.debug("Failed to clean up log file", exc_info=True)
            self._log_file = None

    def stop(self) -> None:
        """서버 프로세스를 중지하고 모든 리소스를 정리합니다."""
        self._stop_process()

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except OSError:
                logger.debug("Failed to clean up temp dir", exc_info=True)
            self._temp_dir = None

        if self._owns_config_dir and self.config_dir is not None:
            import shutil

            try:
                shutil.rmtree(self.config_dir)
            except OSError:
                logger.debug(
                    "Failed to clean up config dir %s", self.config_dir, exc_info=True
                )
            self._owns_config_dir = False

    def update_env(self, **overrides: str) -> None:
        """스테이지 환경 변수는 다음 `restart()`에 적용되도록 재정의됩니다.

        이는 하위 프로세스가 시작되기 직전에 `os.environ`에 적용되며 변형 범위는 다시 시작 호출로 유지됩니다.

        Args:
            **overrides: 키/값 환경 변수 쌍(예:
                         `DEEPAGENTS_CLI_SERVER_MODEL="anthropic:claude-sonnet-4-6"`).

        """
        self._env_overrides.update(overrides)

    async def restart(self, *, timeout: float = _HEALTH_TIMEOUT) -> None:  # noqa: ASYNC109
        """기존 구성 디렉터리를 재사용하여 서버 프로세스를 다시 시작합니다.

        하위 프로세스를 중지한 다음 새 프로세스를 시작합니다. `update_env()`을 통해 준비된 모든 환경 재정의는
        `_scoped_env_overrides` 컨텍스트 관리자 내에 적용되므로 오류가 발생하면 자동으로 환경이 마지막으로 알려진 양호한 상태로
        롤백됩니다.

        Args:
            timeout: 서버가 정상 상태가 될 때까지 기다리는 최대 시간(초)입니다.

        """
        logger.info("Restarting langgraph dev server")
        self._stop_process()

        with _scoped_env_overrides(self._env_overrides):
            await self.start(timeout=timeout)

        self._env_overrides.clear()

    async def __aenter__(self) -> Self:
        """비동기 컨텍스트 관리자 항목.

        Returns:
            서버 프로세스 인스턴스입니다.

        """
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """비동기 컨텍스트 관리자 종료."""
        self.stop()
