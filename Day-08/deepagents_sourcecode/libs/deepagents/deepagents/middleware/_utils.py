"""미들웨어 공통 유틸리티 함수 모듈.

이 모듈은 여러 미들웨어에서 공통으로 사용하는 헬퍼 함수를 제공합니다.
현재는 시스템 메시지에 텍스트를 안전하게 추가하는 `append_to_system_message` 함수만 포함합니다.

핵심 개념:
    - LangChain의 `SystemMessage`는 LLM에게 전달되는 시스템 프롬프트를 담는 메시지 객체입니다.
    - 미들웨어들은 각자 시스템 프롬프트에 자신의 지시사항(도구 사용법, 메모리 내용 등)을
      주입해야 하는데, 이 함수가 그 공통 작업을 담당합니다.
    - `content_blocks` 기반으로 동작하여 기존 콘텐츠를 보존하면서 새 텍스트를 추가합니다.
"""

from langchain_core.messages import ContentBlock, SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """시스템 메시지에 텍스트를 추가합니다.

    기존 시스템 메시지가 있으면 그 뒤에 새 텍스트를 추가하고,
    없으면 새 SystemMessage를 생성합니다.

    여러 미들웨어가 순차적으로 이 함수를 호출하여 시스템 프롬프트를
    점진적으로 조립하는 패턴으로 사용됩니다.

    예시 흐름:
        1. FilesystemMiddleware → "## Filesystem Tools ..." 추가
        2. MemoryMiddleware → "<agent_memory>...</agent_memory>" 추가
        3. SkillsMiddleware → "## Skills System ..." 추가
        → 최종 시스템 메시지에 세 미들웨어의 지시사항이 모두 포함됨

    Args:
        system_message: 기존 시스템 메시지 객체. None이면 빈 상태에서 시작합니다.
        text: 시스템 메시지에 추가할 텍스트 문자열.

    Returns:
        텍스트가 추가된 새로운 SystemMessage 객체.
        (원본 system_message는 변경되지 않는 불변(immutable) 패턴)
    """
    # 기존 시스템 메시지의 content_blocks를 리스트로 복사 (없으면 빈 리스트)
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []

    # 이미 내용이 있으면 빈 줄 2개를 앞에 추가하여 시각적 구분을 만듦
    if new_content:
        text = f"\n\n{text}"

    # 텍스트 블록을 content_blocks 리스트 끝에 추가
    new_content.append({"type": "text", "text": text})

    # 새 SystemMessage 객체를 생성하여 반환 (불변 패턴 유지)
    return SystemMessage(content_blocks=new_content)
