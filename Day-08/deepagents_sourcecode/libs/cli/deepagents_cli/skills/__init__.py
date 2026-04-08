"""deepagents CLI용 기술 모듈.

공용 API: - excute_skills_command: 기술 하위 명령 실행(list/create/info/delete) -
setup_skills_parser: 기술 명령에 대한 인수 구문 분석 구성 설정

다른 모든 구성 요소는 내부 구현 세부 사항입니다.
"""

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
