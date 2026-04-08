"""LangSmith 샌드박스 백엔드 구현체."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from langsmith.sandbox import Sandbox

logger = logging.getLogger(__name__)


class LangSmithSandbox(BaseSandbox):
    """`SandboxBackendProtocol`을 준수하는 LangSmith 샌드박스 구현체.

    이 구현체는 `BaseSandbox`로부터 모든 파일 작업 메서드를 상속하며,
    LangSmith의 API를 사용하여 execute() 메서드만 구현합니다.
    """

    def __init__(self, sandbox: Sandbox) -> None:
        """기존 LangSmith 샌드박스를 래핑하는 백엔드를 생성합니다.

        Args:
            sandbox: 래핑할 LangSmith Sandbox 인스턴스.
        """
        self._sandbox = sandbox
        self._default_timeout: int = 30 * 60

    @property
    def id(self) -> str:
        """LangSmith 샌드박스 이름을 반환합니다."""
        return self._sandbox.name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """샌드박스 내에서 셸 명령을 실행합니다.

        Args:
            command: 실행할 셸 명령 문자열.
            timeout: 명령 완료까지 대기할 최대 시간(초).

                None이면 백엔드의 기본 타임아웃을 사용합니다.
                `langsmith[sandbox]` extra가 설치된 경우 0은 명령 타임아웃을
                비활성화합니다.

        Returns:
            출력, 종료 코드, 절삭 플래그를 포함하는 `ExecuteResponse`.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.run(command, timeout=effective_timeout)

        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr

        return ExecuteResponse(
            output=output,
            exit_code=result.exit_code,
            truncated=False,
        )

    def write(self, file_path: str, content: str) -> WriteResult:
        """ARG_MAX를 피하기 위해 LangSmith SDK를 사용하여 내용을 씁니다.

        `BaseSandbox.write()`는 셸 명령에 전체 내용을 포함하여 전송하므로
        대용량 내용의 경우 ARG_MAX를 초과할 수 있습니다. 이 오버라이드는
        HTTP 본문으로 내용을 전송하는 SDK의 네이티브 `write()`를 사용합니다.

        Args:
            file_path: 샌드박스 내의 대상 경로.
            content: 쓸 텍스트 내용.

        Returns:
            성공 시 기록된 경로, 실패 시 오류 메시지를 포함하는 `WriteResult`.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        try:
            self._sandbox.write(file_path, content.encode("utf-8"))
            return WriteResult(path=file_path)
        except SandboxClientError as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """LangSmith 샌드박스에서 여러 파일을 다운로드합니다.

        부분 성공을 지원합니다 — 개별 다운로드 실패가 다른 파일에 영향을 주지 않습니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로당 하나씩 `FileDownloadResponse` 객체 목록.

                응답 순서는 입력 순서와 일치합니다.
        """
        from langsmith.sandbox import ResourceNotFoundError, SandboxClientError  # noqa: PLC0415

        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not path.startswith("/"):
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                continue
            try:
                content = self._sandbox.read(path)
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except ResourceNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except SandboxClientError as e:
                msg = str(e).lower()
                error = "is_directory" if "is a directory" in msg else "file_not_found"
                responses.append(FileDownloadResponse(path=path, content=None, error=error))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """LangSmith 샌드박스에 여러 파일을 업로드합니다.

        부분 성공을 지원합니다 — 개별 업로드 실패가 다른 파일에 영향을 주지 않습니다.

        Args:
            files: 업로드할 `(path, content)` 튜플 목록.

        Returns:
            입력 파일당 하나씩 `FileUploadResponse` 객체 목록.

                응답 순서는 입력 순서와 일치합니다.
        """
        from langsmith.sandbox import SandboxClientError  # noqa: PLC0415

        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                self._sandbox.write(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except SandboxClientError as e:
                logger.debug("Failed to upload %s: %s", path, e)
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses
