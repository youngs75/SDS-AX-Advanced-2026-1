"""샌드박스 통합에 사용하는 공급자 인터페이스와 공통 오류입니다.

구체적인 공급자 구현은 이 계약을 따르므로, CLI 코드는 특정 벤더 SDK에 직접
의존하지 않고도 샌드박스를 프로비저닝하고 재개하며 삭제할 수 있습니다.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol


class SandboxError(Exception):
    """샌드박스 공급자 작업 전반에서 사용하는 기본 오류입니다."""

    @property
    def original_exc(self) -> BaseException | None:
        """이 오류를 유발한 원본 예외가 있으면 반환합니다."""
        return self.__cause__


class SandboxNotFoundError(SandboxError):
    """요청한 샌드박스를 찾을 수 없을 때 발생합니다."""


class SandboxProvider(ABC):
    """샌드박스 백엔드를 생성하고 삭제하는 공급자 인터페이스입니다."""

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """기존 샌드박스를 가져오거나 필요하면 새로 생성합니다."""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """ID로 샌드박스를 삭제합니다."""
        raise NotImplementedError

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """`get_or_create`를 비동기에서 사용할 수 있도록 감싼 래퍼입니다.

        Returns:
            생성했거나 기존에 존재하던 샌드박스 백엔드입니다.
        """
        return await asyncio.to_thread(
            self.get_or_create, sandbox_id=sandbox_id, **kwargs
        )

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """`delete`를 비동기에서 사용할 수 있도록 감싼 래퍼입니다."""
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)
