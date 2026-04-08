"""`LocalShellBackend`: 제한 없는 로컬 셸 실행 기능을 갖춘 파일시스템 백엔드.

이 백엔드는 FilesystemBackend를 확장하여 로컬 호스트 시스템에서 셸 명령 실행 기능을 추가합니다.
샌드박싱이나 격리를 전혀 제공하지 않으며 — 모든 작업이 전체 시스템 접근 권한으로
호스트 머신에서 직접 실행됩니다.
"""

from __future__ import annotations

import os
import subprocess
import uuid
import warnings
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

if TYPE_CHECKING:
    from pathlib import Path


DEFAULT_EXECUTE_TIMEOUT = 120
"""셸 명령 실행의 기본 타임아웃 시간(초)."""


class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """제한 없는 로컬 셸 명령 실행 기능을 갖춘 파일시스템 백엔드.

    이 백엔드는 `FilesystemBackend`를 확장하여 셸 명령 실행 기능을 추가합니다.
    명령은 샌드박싱, 프로세스 격리, 보안 제한 없이 호스트 시스템에서 직접 실행됩니다.

    !!! warning "보안 경고"

        이 백엔드는 에이전트에게 로컬 머신에 대한 직접 파일시스템 접근과
        제한 없는 셸 실행 권한을 모두 부여합니다. 극도로 주의하여 사용하고
        적절한 환경에서만 사용하십시오.

        **적절한 사용 사례:**

        - 로컬 개발 CLI (코딩 도우미, 개발 도구)
        - 에이전트의 코드를 신뢰하는 개인 개발 환경
        - 적절한 시크릿 관리를 갖춘 CI/CD 파이프라인 (보안 고려사항 참조)

        **부적절한 사용 사례:**

        - 프로덕션 환경 (예: 웹 서버, API, 멀티테넌트 시스템)
        - 신뢰할 수 없는 사용자 입력 처리 또는 신뢰할 수 없는 코드 실행

        프로덕션에는 `StateBackend`, `StoreBackend` 또는 `BaseSandbox` 확장을 사용하십시오.

        **보안 위험:**

        - 에이전트는 사용자 권한으로 **임의의 셸 명령**을 실행할 수 있음
        - 에이전트는 시크릿(API 키, 자격증명, `.env` 파일, SSH 키 등)을 포함한
            **모든 접근 가능한 파일**을 읽을 수 있음
        - 네트워크 도구와 결합하면 SSRF 공격을 통해 시크릿이 유출될 수 있음
        - 파일 수정 및 명령 실행은 **영구적이고 되돌릴 수 없음**
        - 에이전트는 패키지 설치, 시스템 파일 수정, 프로세스 생성 등을 할 수 있음
        - **프로세스 격리 없음** — 명령이 호스트 시스템에서 직접 실행됨
        - **리소스 제한 없음** — 명령이 무제한 CPU, 메모리, 디스크를 사용할 수 있음

        **권장 안전장치:**

        셸 접근이 제한 없고 파일시스템 제한을 우회할 수 있으므로:

        1. **Human-in-the-Loop (HITL) 미들웨어 활성화** — 실행 전 모든 작업을 검토하고
            승인합니다. 이 백엔드를 사용할 때 기본 안전장치로 **강력히 권장**됩니다.
        2. 전용 개발 환경에서만 실행 — 공유 또는 프로덕션 시스템에서는 절대 사용 금지
        3. 신뢰할 수 없는 사용자에게 노출하거나 신뢰할 수 없는 코드 실행 허용 금지
        4. 코드 실행이 필요한 프로덕션 환경에서는 `BaseSandbox`를 확장하여
            적절히 격리된 백엔드(Docker 컨테이너, VM 또는 기타 샌드박스 실행 환경)를 생성하십시오

        !!! note

            `virtual_mode=True`와 경로 기반 제한은 셸 접근이 활성화된 경우
            보안을 제공하지 않습니다 — 명령이 시스템의 모든 경로에 접근할 수 있기 때문입니다

    Examples:
        ```python
        from deepagents.backends import LocalShellBackend

        # 명시적 환경으로 백엔드 생성
        backend = LocalShellBackend(root_dir="/home/user/project", env={"PATH": "/usr/bin:/bin"})

        # 셸 명령 실행 (호스트에서 직접 실행)
        result = backend.execute("ls -la")
        print(result.output)
        print(result.exit_code)

        # 파일시스템 작업 사용 (FilesystemBackend에서 상속)
        content = backend.read("/README.md")
        backend.write("/output.txt", "Hello world")

        # 모든 환경 변수 상속
        backend = LocalShellBackend(root_dir="/home/user/project", inherit_env=True)
        ```
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool | None = None,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """파일시스템 접근 기능을 갖춘 로컬 셸 백엔드를 초기화합니다.

        Args:
            root_dir: 파일시스템 작업과 셸 명령 모두의 작업 디렉터리.

                - 제공하지 않으면 현재 작업 디렉터리로 기본 설정됩니다.
                - 셸 명령은 이 디렉터리를 작업 디렉터리로 하여 실행됩니다.
                - `virtual_mode=False`(기본값)일 때: 경로가 있는 그대로 사용됩니다.
                    에이전트는 절대 경로나 `..` 시퀀스를 사용하여 모든 파일에 접근할 수 있습니다.
                - `virtual_mode=True`일 때: 파일시스템 작업의 가상 루트 역할을 합니다.
                    다양한 백엔드 구현에 걸쳐 파일 작업 라우팅을 지원하는 `CompositeBackend`와
                    함께 유용합니다. **참고:** 이것은 셸 명령을 제한하지 않습니다.

            virtual_mode: 파일시스템 작업의 가상 경로 모드 활성화.

                `True`이면 `root_dir`을 가상 루트 파일시스템으로 취급합니다.
                모든 경로가 `root_dir`에 상대적으로 해석됩니다(예: `/file.txt`는
                `{root_dir}/file.txt`에 매핑됨). 경로 탐색(`..`, `~`)이 차단됩니다.

                **주요 사용 사례:** 다양한 경로 접두사를 다른 백엔드로 라우팅하는
                `CompositeBackend`와 함께 동작합니다. 가상 모드를 사용하면
                CompositeBackend가 라우트 접두사를 제거하고 각 백엔드에 정규화된
                경로를 전달하여 여러 백엔드 구현에서 파일 작업이 올바르게 동작합니다.

                **중요:** 이것은 파일시스템 작업에만 영향을 미칩니다. `execute()`를
                통해 실행되는 셸 명령은 제한되지 않으며 모든 경로에 접근할 수 있습니다.

            timeout: 셸 명령 실행 대기 기본 최대 시간(초).

                기본값은 120초(2분)입니다.

                이 타임아웃을 초과하는 명령은 종료됩니다.

                `execute()`의 `timeout` 파라미터로 명령별로 오버라이드할 수 있습니다.

            max_output_bytes: 명령 출력에서 캡처할 최대 바이트 수.
                이 제한을 초과하는 출력은 절삭됩니다. 기본값은 100,000 바이트.

            env: 셸 명령의 환경 변수. None이면 빈 환경으로 시작합니다
                (`inherit_env=True`가 아닌 경우).

            inherit_env: 부모 프로세스의 환경 변수를 상속할지 여부.
                False(기본값)이면 `env` 딕셔너리의 변수만 사용 가능합니다.
                True이면 모든 `os.environ` 변수를 상속하고 `env` 오버라이드를 적용합니다.

        Raises:
            ValueError: timeout이 양수가 아닌 경우.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        if virtual_mode is None:
            warnings.warn(
                "LocalShellBackend virtual_mode default will change in deepagents 0.5.0; "
                "please specify virtual_mode explicitly. "
                "Note: virtual_mode is for virtual path semantics (e.g., CompositeBackend routing) and optional path-based guardrails; "
                "it does not provide sandboxing or process isolation. "
                "Security note: leaving virtual_mode=False allows absolute paths and '..' to bypass root_dir, "
                "and LocalShellBackend provides no sandboxing (execute runs commands on the host; virtual_mode does not restrict shell execution). "
                "See https://reference.langchain.com/python/deepagents/ for usage guidelines.",
                DeprecationWarning,
                stacklevel=2,
            )
            virtual_mode = False

        # 부모 FilesystemBackend 초기화
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=10,
        )

        # 실행 파라미터 저장
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes

        # inherit_env 설정에 따라 환경 구성
        if inherit_env:
            self._env = os.environ.copy()
            if env is not None:
                self._env.update(env)
        else:
            self._env = env if env is not None else {}

        # 고유한 샌드박스 ID 생성
        self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """이 백엔드 인스턴스의 고유 식별자.

        Returns:
            "local-{random_hex}" 형식의 문자열 식별자.
        """
        return self._sandbox_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""호스트 시스템에서 직접 셸 명령을 실행합니다.

        !!! danger "제한 없는 실행"

            명령은 `shell=True`로 `subprocess.run()`을 사용하여 호스트 시스템에서
            직접 실행됩니다. **샌드박싱, 격리, 보안 제한이 전혀 없습니다**.
            명령은 사용자의 전체 권한으로 실행되며 다음을 할 수 있습니다:

            - 파일시스템의 모든 파일에 접근(`virtual_mode`와 무관)
            - 모든 프로그램이나 스크립트 실행
            - 네트워크 연결 생성
            - 시스템 구성 수정
            - 추가 프로세스 생성
            - 패키지 설치 또는 의존성 수정

            **이 메서드를 사용할 때는 항상 Human-in-the-Loop (HITL) 미들웨어를 사용하십시오.**

        명령은 백엔드의 `root_dir`을 작업 디렉터리로 하여 시스템 셸(`/bin/sh` 또는 동등한 것)로
        실행됩니다. stdout과 stderr는 단일 출력 스트림으로 결합됩니다.

        Args:
            command: 실행할 셸 명령 문자열.
                예: "python script.py", "ls -la", "grep pattern file.txt"

                **보안:** 이 문자열은 셸에 직접 전달됩니다. 에이전트는
                파이프, 리다이렉트, 명령 치환 등을 포함한 임의의 명령을 실행할 수 있습니다.
            timeout: 이 명령에 대한 최대 대기 시간(초).

                초기화 시 설정한 기본 타임아웃을 오버라이드합니다.

                None이면 기본값을 사용합니다.

        Returns:
            다음을 포함하는 ExecuteResponse:
                - output: 결합된 stdout과 stderr (stderr 줄은 [stderr] 접두사 포함)
                - exit_code: 프로세스 종료 코드 (성공은 0, 실패는 비영값)
                - truncated: 크기 제한으로 출력이 절삭된 경우 True

        Raises:
            ValueError: 명령별 타임아웃이 양수가 아닌 경우.

        Examples:
            ```python
            # 간단한 명령 실행
            result = backend.execute("echo hello")
            assert result.output == "hello\\n"
            assert result.exit_code == 0

            # 오류 처리
            result = backend.execute("cat nonexistent.txt")
            assert result.exit_code != 0
            assert "[stderr]" in result.output

            # 절삭 확인
            result = backend.execute("cat huge_file.txt")
            if result.truncated:
                print("Output was truncated")

            # 장시간 실행 명령에 대한 타임아웃 오버라이드
            result = backend.execute("make build", timeout=300)

            # 명령은 root_dir에서 실행되지만 모든 경로에 접근 가능
            result = backend.execute("cat /etc/passwd")  # 시스템 파일을 읽을 수 있음!
            ```
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            result = subprocess.run(  # noqa: S602
                command,
                check=False,
                shell=True,  # 의도적: LLM 제어 셸 실행을 위해 설계됨
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),  # FilesystemBackend의 root_dir 사용
            )

            # stdout과 stderr 결합
            # 명확한 출처 표시를 위해 각 stderr 줄에 [stderr] 접두사를 붙입니다.
            # 예: "hello\n[stderr] error: file not found"  # noqa: ERA001
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # 절삭 확인
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            # 비영 종료 코드인 경우 정보 추가
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = f"Error: Command timed out after {effective_timeout} seconds (custom timeout). The command may be stuck or require more time."
            else:
                msg = f"Error: Command timed out after {effective_timeout} seconds. For long-running commands, re-run using the timeout parameter."
            return ExecuteResponse(
                output=msg,
                exit_code=124,  # 표준 타임아웃 종료 코드
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            # 광범위한 예외 처리는 의도적: 모든 실행 오류를 포착하고
            # 예외를 전파하는 대신 일관된 ExecuteResponse를 반환하기 위함
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
