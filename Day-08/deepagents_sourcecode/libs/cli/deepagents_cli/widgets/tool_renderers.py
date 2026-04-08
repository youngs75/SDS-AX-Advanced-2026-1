"""특수 승인 위젯 렌더러에 도구 이름을 매핑합니다.

이 모듈의 레지스트리는 인수를 요약하는 방법을 알고 있는 렌더러로 각 도구 호출을 라우팅하여 HITL 미리 보기의 일관성을 유지합니다.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from deepagents_cli.widgets.tool_widgets import (
    EditFileApprovalWidget,
    GenericApprovalWidget,
    WriteFileApprovalWidget,
)

if TYPE_CHECKING:
    from deepagents_cli.widgets.tool_widgets import ToolApprovalWidget


class ToolRenderer:
    """도구의 HITL 승인 위젯을 구축하기 위한 전략입니다.

    각 렌더러는 승인 상자에서 사용자에게 표시되는 내용을 제어하는 ​​`(widget_class, data)` 쌍에 도구 이름을 매핑합니다.
    `_RENDERER_REGISTRY`에 등록되지 않은 도구는 기본값으로 변경되어 모든 인수를 `GenericApprovalWidget`을 통해
    `key: value` 줄로 덤프합니다.

    """

    @staticmethod
    def get_approval_widget(
        tool_args: dict[str, Any],
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        """이 도구에 대한 승인 위젯 클래스와 데이터를 가져옵니다.

        Args:
            tool_args: action_request의 도구 인수

        Returns:
            (widget_class, data_dict)의 튜플

        """
        return GenericApprovalWidget, tool_args


class WriteFileRenderer(ToolRenderer):
    """write_file 도구용 렌더러 - 전체 파일 내용을 표시합니다."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        # Extract file extension for syntax highlighting
        file_path = tool_args.get("file_path", "")
        content = tool_args.get("content", "")

        # Get file extension
        file_extension = "text"
        if "." in file_path:
            file_extension = file_path.rsplit(".", 1)[-1]

        data = {
            "file_path": file_path,
            "content": content,
            "file_extension": file_extension,
        }
        return WriteFileApprovalWidget, data


class TaskRenderer(ToolRenderer):
    """작업 도구용 렌더러입니다. 인터럽트 설명이 전체 컨텍스트를 제공합니다."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],  # noqa: ARG004  # Unused; interrupt description already formats task args
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        return GenericApprovalWidget, {}


class EditFileRenderer(ToolRenderer):
    """edit_file 도구용 렌더러 - 통합 diff를 표시합니다."""

    @staticmethod
    def get_approval_widget(  # noqa: D102  # Protocol method — docstring on base class
        tool_args: dict[str, Any],
    ) -> tuple[type[ToolApprovalWidget], dict[str, Any]]:
        file_path = tool_args.get("file_path", "")
        old_string = tool_args.get("old_string", "")
        new_string = tool_args.get("new_string", "")

        # Generate unified diff
        diff_lines = EditFileRenderer._generate_diff(old_string, new_string)

        data = {
            "file_path": file_path,
            "diff_lines": diff_lines,
            "old_string": old_string,
            "new_string": new_string,
        }
        return EditFileApprovalWidget, data

    @staticmethod
    def _generate_diff(old_string: str, new_string: str) -> list[str]:
        """이전 문자열과 새 문자열에서 통합된 diff 라인을 생성합니다.

        Returns:
            파일 헤더가 없는 diff 줄 목록입니다.

        """
        if not old_string and not new_string:
            return []

        old_lines = old_string.split("\n") if old_string else []
        new_lines = new_string.split("\n") if new_string else []

        # Generate unified diff
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=3,  # Context lines
        )

        # Skip the first two header lines (--- and +++)
        diff_list = list(diff)
        return diff_list[2:] if len(diff_list) > 2 else diff_list  # noqa: PLR2004  # Column count threshold


_RENDERER_REGISTRY: dict[str, type[ToolRenderer]] = {
    "task": TaskRenderer,
    "write_file": WriteFileRenderer,
    "edit_file": EditFileRenderer,
}
"""렌더러에 대한 레지스트리 매핑 도구 이름

참고: bash/shell/execute는 최소 승인을 사용합니다(렌더러 없음). ApprovalMenu._MINIMAL_TOOLS를 참조하세요.
"""


def get_renderer(tool_name: str) -> ToolRenderer:
    """도구의 렌더러를 이름으로 가져옵니다.

    Args:
        tool_name: 도구 이름

    Returns:
        적절한 ToolRenderer 인스턴스

    """
    renderer_class = _RENDERER_REGISTRY.get(tool_name, ToolRenderer)
    return renderer_class()
