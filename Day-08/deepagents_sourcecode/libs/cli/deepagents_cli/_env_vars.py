"""`DEEPAGENTS_CLI_*` 환경 변수의 정식 레지스트리입니다.

이름이 `DEEPAGENTS_CLI_`으로 시작하는 CLI가 읽는 모든 env var는 여기에서 모듈 수준 상수로 정의되어야 합니다.  이 모듈에서 가져온
상수 대신 `"DEEPAGENTS_CLI_FOO"`와 같은 순수 문자열 리터럴이 소스 코드에 나타나면 드리프트 감지
테스트(`tests/unit_tests/test_env_vars.py`)가 실패합니다.

원시 문자열 리터럴을 사용하는 대신 짧은 이름 상수(예: `AUTO_UPDATE`, `DEBUG`)를 가져와 `os.environ.get()`에 전달합니다.
env var의 이름이 바뀌면 여기의 값만 변경됩니다.

!!! 메모

    `resolve_env_var`은(는) API 키 및 공급자 자격 증명에 대한 동적 접두사 재정의도 지원합니다.
    `DEEPAGENTS_CLI_{NAME}` 설정은 `{NAME}`보다 우선합니다.  예를 들어
    `DEEPAGENTS_CLI_OPENAI_API_KEY`은 `OPENAI_API_KEY`을 재정의합니다. `resolve_env_var`을 사용하는
    호출 사이트만 이 이점을 누릴 수 있습니다. 직접 `os.environ.get` 조회(아래 상수와 같은)는 그렇지 않습니다. 동적 재정의는 타사 변수
    이름을 미러링하므로 여기에 나열되지 않습니다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 상수 — 순수 문자열 리터럴 대신 이를 가져옵니다.
# 상수 이름을 기준으로 알파벳순으로 정렬됩니다.
# ---------------------------------------------------------------------------

AUTO_UPDATE = "DEEPAGENTS_CLI_AUTO_UPDATE"
"""자동 CLI 업데이트를 활성화합니다('1', 'true' 또는 'yes')."""

DEBUG = "DEEPAGENTS_CLI_DEBUG"
"""파일에 대한 자세한 디버그 로깅을 활성화합니다."""

DEBUG_FILE = "DEEPAGENTS_CLI_DEBUG_FILE"
"""디버그 로그 파일의 경로(기본값: `/tmp/deepagents_debug.log`)."""

EXTRA_SKILLS_DIRS = "DEEPAGENTS_CLI_EXTRA_SKILLS_DIRS"
"""스킬 격리 허용 목록에 콜론으로 구분된 경로가 추가되었습니다."""

LANGSMITH_PROJECT = "DEEPAGENTS_CLI_LANGSMITH_PROJECT"
"""에이전트 추적을 위해 LangSmith 프로젝트 이름을 재정의합니다."""

NO_UPDATE_CHECK = "DEEPAGENTS_CLI_NO_UPDATE_CHECK"
"""설정 시 자동 업데이트 확인을 비활성화합니다."""

SERVER_ENV_PREFIX = "DEEPAGENTS_CLI_SERVER_"
"""CLI 구성을 서버 하위 프로세스에 전달하는 데 사용되는 환경 변수 접두사입니다."""

SHELL_ALLOW_LIST = "DEEPAGENTS_CLI_SHELL_ALLOW_LIST"
"""허용(또는 '권장'/'모두')할 쉼표로 구분된 셸 명령입니다."""

USER_ID = "DEEPAGENTS_CLI_USER_ID"
"""LangSmith 추적 메타데이터에 사용자 식별자를 연결합니다."""
