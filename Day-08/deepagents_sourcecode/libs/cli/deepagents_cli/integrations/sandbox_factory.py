"""선택적 샌드박스 백엔드를 생성, 구성 및 해체합니다.

이 모듈은 설치된 샌드박스 공급자를 검색하고 공유 CLI 설정을 적용하며 패키지의 나머지 부분에 균일한 수명 주기 API를 제공합니다.
"""


from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import shlex
import string
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape as escape_markup

from deepagents_cli.config import console, get_glyphs
from deepagents_cli.integrations.sandbox_provider import (
    SandboxNotFoundError,
    SandboxProvider,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from deepagents.backends.protocol import SandboxBackendProtocol
    from langsmith.sandbox import SandboxTemplate


def _run_sandbox_setup(backend: SandboxBackendProtocol, setup_script_path: str) -> None:
    """env var 확장을 사용하여 샌드박스에서 사용자 설정 스크립트를 실행합니다.

    Args:
        backend: 샌드박스 백엔드 인스턴스
        setup_script_path: 설정 스크립트 파일 경로

    Raises:
        FileNotFoundError: 설정 스크립트가 존재하지 않는 경우.
        RuntimeError: 설정 스크립트가 실행되지 않는 경우.

    """

    script_path = Path(setup_script_path)
    if not script_path.exists():
        msg = f"Setup script not found: {setup_script_path}"
        raise FileNotFoundError(msg)

    console.print(
        f"[dim]Running setup script: {escape_markup(setup_script_path)}...[/dim]"
    )

    # Read script content
    script_content = script_path.read_text(encoding="utf-8")

    # Expand ${VAR} syntax using local environment
    template = string.Template(script_content)
    expanded_script = template.safe_substitute(os.environ)

    # Execute expanded script in sandbox
    result = backend.execute(f"bash -c {shlex.quote(expanded_script)}")

    if result.exit_code != 0:
        console.print(f"[red]Setup script failed (exit {result.exit_code}):[/red]")
        console.print(f"[dim]{escape_markup(result.output)}[/dim]")
        msg = "Setup failed - aborting"
        raise RuntimeError(msg)

    console.print(f"[green]{get_glyphs().checkmark} Setup complete[/green]")


_PROVIDER_TO_WORKING_DIR = {
    "agentcore": "/tmp",  # noqa: S108 # AgentCore Code Interpreter working directory
    "daytona": "/home/daytona",
    "langsmith": "/tmp",  # noqa: S108  # LangSmith sandbox working directory
    "modal": "/workspace",
    "runloop": "/home/user",
}
"""샌드박스 공급자 이름을 기본 작업 디렉터리에 매핑합니다."""



@contextmanager
def create_sandbox(
    provider: str,
    *,
    sandbox_id: str | None = None,
    setup_script_path: str | None = None,
) -> Generator[SandboxBackendProtocol, None, None]:
    """지정된 공급자의 샌드박스를 생성하거나 연결합니다.

    이는 공급자 추상화를 사용하여 샌드박스 생성을 위한 통합 인터페이스입니다.

    Args:
        provider: 샌드박스 제공업체(`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`,
                  `'runloop'`)
        sandbox_id: 재사용할 선택적 기존 샌드박스 ID
        setup_script_path: 샌드박스가 시작된 후 실행할 설정 스크립트의 선택적 경로

    Yields:
        `SandboxBackendProtocol` 인스턴스

    """

    # Get provider instance
    provider_obj = _get_provider(provider)

    # Determine if we should cleanup (only cleanup if we created it)
    should_cleanup = sandbox_id is None

    # Create or connect to sandbox
    console.print(f"[yellow]Starting {provider} sandbox...[/yellow]")
    backend = provider_obj.get_or_create(sandbox_id=sandbox_id)
    glyphs = get_glyphs()
    console.print(
        f"[green]{glyphs.checkmark} {provider.capitalize()} sandbox ready: "
        f"{backend.id}[/green]"
    )

    # Run setup script if provided
    if setup_script_path:
        _run_sandbox_setup(backend, setup_script_path)

    try:
        yield backend
    finally:
        if should_cleanup:
            try:
                console.print(
                    f"[dim]Terminating {provider} sandbox {backend.id}...[/dim]"
                )
                provider_obj.delete(sandbox_id=backend.id)
                glyphs = get_glyphs()
                console.print(
                    f"[dim]{glyphs.checkmark} {provider.capitalize()} sandbox "
                    f"{backend.id} terminated[/dim]"
                )
            except Exception as e:  # noqa: BLE001  # Cleanup errors should not mask the original sandbox failure
                warning = get_glyphs().warning
                console.print(
                    f"[yellow]{warning} Cleanup failed for {provider} sandbox "
                    f"{backend.id}: {e}[/yellow]"
                )


def _get_available_sandbox_types() -> list[str]:
    """사용 가능한 샌드박스 공급자 유형 목록을 가져옵니다(내부).

    Returns:
        사용 가능한 샌드박스 공급자 유형 이름 목록

    """

    return sorted(_PROVIDER_TO_WORKING_DIR.keys())


def get_default_working_dir(provider: str) -> str:
    """특정 샌드박스 공급자의 기본 작업 디렉터리를 가져옵니다.

    Args:
        provider: 샌드박스 공급자 이름(`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`,
                  `'runloop'`)

    Returns:
        기본 작업 디렉터리 경로(문자열)

    Raises:
        ValueError: 공급자를 알 수 없는 경우

    """

    if provider in _PROVIDER_TO_WORKING_DIR:
        return _PROVIDER_TO_WORKING_DIR[provider]
    msg = f"Unknown sandbox provider: {provider}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------


def _import_provider_module(
    module_name: str,
    *,
    provider: str,
    package: str,
) -> ModuleType:
    """공급자별 오류 메시지와 함께 선택적 공급자 모듈을 가져옵니다.

    Args:
        module_name: 가져올 Python 모듈 이름입니다.
        provider: 샌드박스 제공자 이름(예: `'daytona'`).
        package: CLI 추가 항목에 의해 노출되는 PyPI 패키지 이름입니다.

    Returns:
        가져온 모듈 개체입니다.

    Raises:
        ImportError: 선택적 종속성이 설치되지 않은 경우.

    """

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        msg = (
            f"The '{provider}' sandbox provider requires the '{package}' package. "
            f"Install it with: pip install 'deepagents-cli[{provider}]'"
        )
        raise ImportError(msg) from exc


_LANGSMITH_DEFAULT_TEMPLATE = "deepagents-cli"
"""템플릿이 지정되지 않은 경우 사용되는 기본 LangSmith 샌드박스 템플릿 이름입니다."""


_LANGSMITH_DEFAULT_IMAGE = "python:3"
"""이미지가 제공되지 않은 경우 LangSmith 샌드박스의 기본 Docker 이미지입니다."""



class _LangSmithProvider(SandboxProvider):
    """LangSmith 샌드박스 공급자 구현.

    LangSmith SDK를 사용하여 LangSmith 샌드박스 수명주기를 관리합니다.

    """


    def __init__(self, api_key: str | None = None) -> None:
        """LangSmith 공급자를 초기화합니다.

        Args:
            api_key: LangSmith API 키(기본값은 `LANGSMITH_SANDBOX_API_KEY`, 그 다음은
                     `LANGSMITH_API_KEY` env var).

        Raises:
            ValueError: LangSmith API 키가 발견되지 않은 경우.

        """

        from langsmith.sandbox import SandboxClient

        from deepagents_cli.model_config import resolve_env_var

        self._api_key = (
            api_key
            or resolve_env_var("LANGSMITH_SANDBOX_API_KEY")
            or resolve_env_var("LANGSMITH_API_KEY")
        )
        if not self._api_key:
            msg = (
                "No LangSmith sandbox API key found. Set "
                "LANGSMITH_SANDBOX_API_KEY or LANGSMITH_API_KEY "
                "(or the DEEPAGENTS_CLI_-prefixed equivalents)."
            )
            raise ValueError(msg)
        self._client: SandboxClient = SandboxClient(api_key=self._api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        template: str | None = None,
        template_image: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """기존 제품을 가져오거나 새로운 LangSmith 샌드박스를 만드세요.

        Args:
            sandbox_id: 재사용할 선택적 기존 샌드박스 이름
            timeout: 샌드박스 시작 시간 초과(초)
            template: 샌드박스의 템플릿 이름
            template_image: 템플릿의 Docker 이미지
            **kwargs: 추가 LangSmith 관련 매개변수

        Returns:
            `LangSmithSandbox` 인스턴스

        Raises:
            RuntimeError: 샌드박스 연결 또는 시작이 실패하는 경우
            TypeError: 지원되지 않는 키워드 인수가 제공된 경우

        """

        from deepagents.backends.langsmith import LangSmithSandbox

        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)
        if sandbox_id:
            # Connect to existing sandbox by name
            try:
                sandbox = self._client.get_sandbox(name=sandbox_id)
            except Exception as e:
                msg = f"Failed to connect to existing sandbox '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e
            return LangSmithSandbox(sandbox)

        resolved_template_name, resolved_image_name = self._resolve_template(
            template, template_image
        )

        # Create new sandbox - ensure template exists first
        self._ensure_template(resolved_template_name, resolved_image_name)

        try:
            sandbox = self._client.create_sandbox(
                template_name=resolved_template_name, timeout=timeout
            )
        except Exception as e:
            msg = (
                f"Failed to create sandbox from template "
                f"'{resolved_template_name}': {e}"
            )
            raise RuntimeError(msg) from e

        # Verify sandbox is ready by polling
        for _ in range(timeout // 2):
            try:
                result = sandbox.run("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception:  # noqa: S110, BLE001  # Sandbox not ready yet, continue polling
                pass
            time.sleep(2)
        else:
            # Cleanup on failure
            with contextlib.suppress(Exception):
                self._client.delete_sandbox(sandbox.name)
            msg = f"LangSmith sandbox failed to start within {timeout} seconds"
            raise RuntimeError(msg)

        return LangSmithSandbox(sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # Required by SandboxFactory interface
        """LangSmith 샌드박스를 삭제합니다.

        Args:
            sandbox_id: 삭제할 샌드박스 이름
            **kwargs: 추가 매개변수

        """

        self._client.delete_sandbox(sandbox_id)

    @staticmethod
    def _resolve_template(
        template: SandboxTemplate | str | None,
        template_image: str | None = None,
    ) -> tuple[str, str]:
        """kwargs에서 템플릿 이름과 이미지를 확인합니다.

        Returns:
            `(template_name, template_image)`의 튜플입니다.

                제공되지 않은 경우 기본값을 사용하여 항상 값을 반환합니다.

        """

        resolved_image = template_image or _LANGSMITH_DEFAULT_IMAGE
        if template is None:
            return _LANGSMITH_DEFAULT_TEMPLATE, resolved_image
        if isinstance(template, str):
            return template, resolved_image
        # SandboxTemplate object - extract image if not provided
        if template_image is None and template.image:
            resolved_image = template.image
        return template.name, resolved_image

    def _ensure_template(
        self,
        template_name: str,
        template_image: str,
    ) -> None:
        """템플릿이 있는지 확인하고 필요한 경우 템플릿을 만듭니다.

        Raises:
            RuntimeError: 템플릿 확인 또는 생성에 실패한 경우

        """

        from langsmith.sandbox import ResourceNotFoundError

        try:
            self._client.get_template(template_name)
        except ResourceNotFoundError as e:
            if e.resource_type != "template":
                msg = f"Unexpected resource not found: {e}"
                raise RuntimeError(msg) from e
            # Template doesn't exist, create it
            try:
                self._client.create_template(name=template_name, image=template_image)
            except Exception as create_err:
                msg = f"Failed to create template '{template_name}': {create_err}"
                raise RuntimeError(msg) from create_err
        except Exception as e:
            msg = f"Failed to check template '{template_name}': {e}"
            raise RuntimeError(msg) from e


class _DaytonaProvider(SandboxProvider):
    """Daytona 샌드박스 공급자 — Daytona 샌드박스의 수명 주기 관리."""
    def __init__(self) -> None:
        daytona_module = _import_provider_module(
            "daytona",
            provider="daytona",
            package="langchain-daytona",
        )

        from deepagents_cli.model_config import resolve_env_var

        api_key = resolve_env_var("DAYTONA_API_KEY")
        if not api_key:
            msg = (
                "No Daytona API key found. Set DAYTONA_API_KEY "
                "or DEEPAGENTS_CLI_DAYTONA_API_KEY."
            )
            raise ValueError(msg)
        self._client = daytona_module.Daytona(
            daytona_module.DaytonaConfig(
                api_key=api_key,
                api_url=resolve_env_var("DAYTONA_API_URL"),
            )
        )

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Daytona 샌드박스를 얻거나 만듭니다.

        Args:
            sandbox_id: 아직 지원되지 않습니다. 없음이어야 합니다.
            timeout: 시작을 기다리는 데 몇 초가 소요됩니다.
            **kwargs: 미사용.

        Returns:
            `DaytonaSandbox` 인스턴스.

        Raises:
            NotImplementedError: `sandbox_id`이 제공된 경우.
            RuntimeError: 샌드박스가 시작되지 않는 경우.

        """

        daytona_backend = _import_provider_module(
            "langchain_daytona",
            provider="daytona",
            package="langchain-daytona",
        )

        if sandbox_id:
            msg = (
                "Connecting to existing Daytona sandbox by ID not yet supported. "
                "Create a new sandbox by omitting sandbox_id parameter."
            )
            raise NotImplementedError(msg)

        sandbox = self._client.create()
        last_exc: Exception | None = None
        for _ in range(timeout // 2):
            try:
                result = sandbox.process.exec("echo ready", timeout=5)
                if result.exit_code == 0:
                    break
            except Exception as exc:  # noqa: BLE001  # Transient failures expected during readiness polling
                last_exc = exc
            time.sleep(2)
        else:
            with contextlib.suppress(Exception):  # Best-effort cleanup
                sandbox.delete()
            detail = f" Last error: {last_exc}" if last_exc else ""
            msg = f"Daytona sandbox failed to start within {timeout} seconds.{detail}"
            raise RuntimeError(msg)

        return daytona_backend.DaytonaSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """ID별로 Daytona 샌드박스를 삭제합니다."""

        sandbox = self._client.get(sandbox_id)
        self._client.delete(sandbox)


class _ModalProvider(SandboxProvider):
    """모달 샌드박스 공급자 — 모달 샌드박스의 수명 주기 관리."""
    def __init__(self) -> None:
        self._modal = _import_provider_module(
            "modal",
            provider="modal",
            package="langchain-modal",
        )

        from deepagents_cli.model_config import resolve_env_var

        token_id = resolve_env_var("MODAL_TOKEN_ID")
        token_secret = resolve_env_var("MODAL_TOKEN_SECRET")
        if token_id and token_secret:
            try:
                self._client = self._modal.Client.from_credentials(
                    token_id, token_secret
                )
            except Exception as exc:
                msg = (
                    "Failed to authenticate with Modal using "
                    "MODAL_TOKEN_ID / MODAL_TOKEN_SECRET "
                    "(or the DEEPAGENTS_CLI_-prefixed equivalents). "
                    "Verify your credentials are valid."
                )
                raise ValueError(msg) from exc
        elif token_id or token_secret:
            logger.warning(
                "Only one of MODAL_TOKEN_ID / MODAL_TOKEN_SECRET is set; "
                "both are required for explicit credential auth. "
                "Falling back to default Modal authentication.",
            )
            self._client = None
        else:
            self._client = None

        lookup_kwargs: dict[str, Any] = {
            "name": "deepagents-sandbox",
            "create_if_missing": True,
        }
        if self._client is not None:
            lookup_kwargs["client"] = self._client
        self._app = self._modal.App.lookup(**lookup_kwargs)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Modal 샌드박스를 가져오거나 만듭니다.

        Args:
            sandbox_id: 기존 샌드박스 ID 또는 생성할 없음입니다.
            timeout: 시작을 기다리는 데 몇 초가 소요됩니다.
            **kwargs: 미사용.

        Returns:
            `ModalSandbox` 인스턴스.

        Raises:
            RuntimeError: 샌드박스가 시작되지 않는 경우.

        """

        modal_backend = _import_provider_module(
            "langchain_modal",
            provider="modal",
            package="langchain-modal",
        )

        client_kwargs: dict[str, Any] = {}
        if self._client is not None:
            client_kwargs["client"] = self._client

        if sandbox_id:
            sandbox = self._modal.Sandbox.from_id(
                sandbox_id=sandbox_id,
                app=self._app,
                **client_kwargs,
            )
        else:
            sandbox = self._modal.Sandbox.create(
                app=self._app, workdir="/workspace", **client_kwargs
            )
            last_exc: Exception | None = None
            for _ in range(timeout // 2):
                if sandbox.poll() is not None:
                    msg = "Modal sandbox terminated unexpectedly during startup"
                    raise RuntimeError(msg)
                try:
                    process = sandbox.exec("echo", "ready", timeout=5)
                    process.wait()
                    if process.returncode == 0:
                        break
                except Exception as exc:  # noqa: BLE001  # Transient failures expected during readiness polling
                    last_exc = exc
                time.sleep(2)
            else:
                sandbox.terminate()
                detail = f" Last error: {last_exc}" if last_exc else ""
                msg = f"Modal sandbox failed to start within {timeout} seconds.{detail}"
                raise RuntimeError(msg)

        return modal_backend.ModalSandbox(sandbox=sandbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """ID로 모달 샌드박스를 종료합니다."""

        del_kwargs: dict[str, Any] = {"sandbox_id": sandbox_id, "app": self._app}
        if self._client is not None:
            del_kwargs["client"] = self._client
        sandbox = self._modal.Sandbox.from_id(**del_kwargs)
        sandbox.terminate()


class _RunloopProvider(SandboxProvider):
    """Runloop 샌드박스 공급자 — Runloop devbox의 수명 주기 관리."""
    def __init__(self) -> None:
        runloop_module = _import_provider_module(
            "runloop_api_client",
            provider="runloop",
            package="langchain-runloop",
        )

        from deepagents_cli.model_config import resolve_env_var

        api_key = resolve_env_var("RUNLOOP_API_KEY")
        if not api_key:
            msg = (
                "No Runloop API key found. Set RUNLOOP_API_KEY "
                "or DEEPAGENTS_CLI_RUNLOOP_API_KEY."
            )
            raise ValueError(msg)
        self._client = runloop_module.Runloop(bearer_token=api_key)

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        timeout: int = 180,
        **kwargs: Any,  # noqa: ARG002
    ) -> SandboxBackendProtocol:
        """Runloop devbox를 얻거나 만드십시오.

        Args:
            sandbox_id: 기존 devbox ID, 또는 생성할 없음.
            timeout: 시작을 기다리는 데 몇 초가 소요됩니다.
            **kwargs: 미사용.

        Returns:
            `RunloopSandbox` 인스턴스.

        Raises:
            RuntimeError: devbox가 시작되지 않는 경우.
            SandboxNotFoundError: `sandbox_id`이 존재하지 않는 경우.

        """

        runloop_backend = _import_provider_module(
            "langchain_runloop",
            provider="runloop",
            package="langchain-runloop",
        )
        runloop_sdk = _import_provider_module(
            "runloop_api_client.sdk",
            provider="runloop",
            package="langchain-runloop",
        )

        if sandbox_id:
            try:
                self._client.devboxes.retrieve(id=sandbox_id)
            except KeyError as e:
                raise SandboxNotFoundError(sandbox_id) from e
        else:
            view = self._client.devboxes.create()
            sandbox_id = view.id
            for _ in range(timeout // 2):
                status = self._client.devboxes.retrieve(id=sandbox_id)
                if status.status == "running":
                    break
                time.sleep(2)
            else:
                self._client.devboxes.shutdown(id=sandbox_id)
                msg = f"Devbox failed to start within {timeout} seconds"
                raise RuntimeError(msg)

        devbox = runloop_sdk.Devbox(self._client, sandbox_id)
        return runloop_backend.RunloopSandbox(devbox=devbox)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002
        """ID로 Runloop devbox를 종료합니다."""

        self._client.devboxes.shutdown(id=sandbox_id)


class _AgentCoreProvider(SandboxProvider):
    """AgentCore 코드 해석기 샌드박스 제공자.

    AgentCore 세션 수명주기를 관리합니다. CLI 종료 후 세션을 다시 연결할 수 없습니다. `sandbox_id` 매개변수는 지원되지 않습니다.

    """


    def __init__(self, region: str | None = None) -> None:
        """AgentCore 공급자를 초기화합니다.

        Args:
            region: AWS 리전(기본값: `AWS_REGION` / `AWS_DEFAULT_REGION` / `us-west-2`)

        Raises:
            ValueError: boto3가 설치되어 있고 AWS 자격 증명을 확인할 수 없는 경우.

        """

        self._region = region or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )

        # Validate AWS credentials early for a clear error message.
        try:
            import boto3  # ty: ignore[unresolved-import]

            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                msg = (
                    "AWS credentials not found. Configure via "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN, "
                    "~/.aws/credentials, or an IAM role."
                )
                raise ValueError(msg)  # noqa: TRY301  # intentional raise for early credential validation
        except ImportError:
            logger.debug("boto3 not installed; skipping credential pre-check")
        except ValueError:
            raise
        except Exception:
            logger.warning(
                "AWS credential pre-validation failed — the session may "
                "fail to start. Check your AWS configuration.",
                exc_info=True,
            )

        self._active_interpreters: dict[str, Any] = {}

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,  # noqa: ARG002  # required by SandboxProvider interface
    ) -> SandboxBackendProtocol:
        """새 AgentCore 코드 해석기 세션을 생성합니다.

        Args:
            sandbox_id: 지원되지 않음 - 제공된 경우 `NotImplementedError`을 발생시킵니다.
            **kwargs: 추가 매개변수(사용되지 않음)

        Returns:
            `AgentCoreSandbox` 시작된 인터프리터를 래핑하는 인스턴스입니다.

        Raises:
            NotImplementedError: `sandbox_id`이 제공된 경우.

        """

        if sandbox_id:
            msg = (
                "AgentCore does not support reconnecting to existing sessions. "
                "Remove the --sandbox-id option."
            )
            raise NotImplementedError(msg)

        agentcore_module = _import_provider_module(
            "bedrock_agentcore.tools.code_interpreter_client",
            provider="agentcore",
            package="langchain-agentcore-codeinterpreter",
        )
        agentcore_backend = _import_provider_module(
            "langchain_agentcore_codeinterpreter",
            provider="agentcore",
            package="langchain-agentcore-codeinterpreter",
        )

        interpreter = agentcore_module.CodeInterpreter(
            region=self._region,
            integration_source="deepagents-cli",
        )
        try:
            interpreter.start()
        except Exception:
            with contextlib.suppress(Exception):
                interpreter.stop()
            raise

        backend = agentcore_backend.AgentCoreSandbox(interpreter=interpreter)
        self._active_interpreters[backend.id] = interpreter
        return backend

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:  # noqa: ARG002  # required by SandboxProvider interface
        """AgentCore 세션을 중지합니다.

        Args:
            sandbox_id: 중지할 세션 ID입니다.
            **kwargs: 추가 매개변수(사용되지 않음)

        """

        interpreter = self._active_interpreters.pop(sandbox_id, None)
        if interpreter:
            try:
                interpreter.stop()
                logger.info("AgentCore session %s stopped", sandbox_id)
            except Exception:
                logger.warning(
                    "Failed to stop AgentCore session %s — the session may "
                    "still be running and incurring costs. Check the AWS "
                    "console to verify.",
                    sandbox_id,
                    exc_info=True,
                )
        else:
            logger.info(
                "AgentCore session %s not tracked (may have already expired)",
                sandbox_id,
            )


def _get_provider(provider_name: str) -> SandboxProvider:
    """지정된 공급자에 대한 `SandboxProvider` 인스턴스를 가져옵니다(내부).

    Args:
        provider_name: 제공자 이름(`'agentcore'`, `'daytona'`, `'langsmith'`, `'modal'`,
                       `'runloop'`)

    Returns:
        `SandboxProvider` 인스턴스

    Raises:
        ValueError: `provider_name`을(를) 알 수 없는 경우입니다.

    """

    if provider_name == "agentcore":
        return _AgentCoreProvider()
    if provider_name == "daytona":
        return _DaytonaProvider()
    if provider_name == "langsmith":
        return _LangSmithProvider()
    if provider_name == "modal":
        return _ModalProvider()
    if provider_name == "runloop":
        return _RunloopProvider()
    msg = (
        f"Unknown sandbox provider: {provider_name}. "
        f"Available providers: {', '.join(_get_available_sandbox_types())}"
    )
    raise ValueError(msg)


def verify_sandbox_deps(provider: str) -> None:
    """샌드박스 공급자에 필요한 패키지가 설치되어 있는지 확인하세요.

    실제 가져오기 없이 간단한 검사를 위해 `importlib.util.find_spec`을 사용합니다. 서버 하위 프로세스를 생성하기 *전에* CLI
    프로세스에서 이를 호출하면 사용자가 불투명한 서버 충돌 대신 명확하고 실행 가능한 오류를 얻을 수 있습니다.

    Args:
        provider: 샌드박스 제공자 이름(예: `'daytona'`).

    Raises:
        ImportError: 공급자의 백엔드 패키지가 설치되지 않은 경우.

    """

    if not provider or provider in {"none", "langsmith"}:
        return

    # Map provider name → (backend module, pip extra).
    # Only the backend module is checked because the underlying SDK is a
    # transitive dependency of the backend package.
    backend_modules: dict[str, tuple[str, str]] = {
        "agentcore": ("langchain_agentcore_codeinterpreter", "agentcore"),
        "daytona": ("langchain_daytona", "daytona"),
        "modal": ("langchain_modal", "modal"),
        "runloop": ("langchain_runloop", "runloop"),
    }

    entry = backend_modules.get(provider)
    if entry is None:
        logger.debug(
            "No backend_modules entry for provider %r; skipping pre-flight check",
            provider,
        )
        return

    module_name, extra = entry
    try:
        found = importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        found = False

    if not found:
        msg = (
            f"Missing dependencies for '{provider}' sandbox. "
            f"Install with: pip install 'deepagents-cli[{extra}]'"
        )
        raise ImportError(msg)


__all__ = [
    "create_sandbox",
    "get_default_working_dir",
    "verify_sandbox_deps",
]
