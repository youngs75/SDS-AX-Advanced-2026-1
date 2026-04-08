"""경량 세션 통계 및 토큰 형식화 유틸리티.

이 모듈은 의도적으로 무거운 종속성(pydantic 없음, 구성 없음, 위젯 가져오기 없음)이 없어 `app.py`이 전체 `textual_adapter`
종속성 트리를 가져오지 않고도 모듈 수준에서 `SessionStats` 및 `format_token_count`을 가져올 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpinnerStatus = Literal["Thinking", "Offloading"] | None
"""유효한 스피너 표시 상태이거나 숨기려면 `None`입니다."""


@dataclass
class ModelStats:
    """세션 내의 단일 모델에 대한 토큰 통계입니다.

Attributes:
        request_count: 이 모델에 대한 LLM API 요청 수입니다.
        input_tokens: 이 모델로 전송된 누적 입력 토큰입니다.
        output_tokens: 이 모델에서 받은 누적 출력 토큰입니다.

    """

    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SessionStats:
    """단일 에이전트 차례(또는 전체 세션)에 걸쳐 누적된 통계입니다.

Attributes:
        request_count: 이루어진 총 LLM API 요청(usage_metadata가 포함된 각 청크는 완료된 요청 1개로 계산됩니다).
        input_tokens: 모든 LLM 요청에 대한 누적 입력 토큰입니다.
        output_tokens: 모든 LLM 요청에 대한 누적 출력 토큰입니다.
        wall_time_seconds: 스트림 시작부터 끝까지의 벽시계 지속 시간입니다.
        per_model: 모델 이름을 기준으로 모델별 분석이 이루어집니다. `record_request`이 비어 있지 않은
                   `model_name`을(를) 수신하는 경우에만 채워집니다. 빈 dict는 명명된 모델 요청이 기록되지 않았음을 의미합니다.
                   `print_usage_table`는 이 경우 모델 테이블을 생략하고 벽 시간 선만 표시합니다(해당하는 경우).

    """

    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    per_model: dict[str, ModelStats] = field(default_factory=dict)

    def record_request(
        self,
        model_name: str,
        input_toks: int,
        output_toks: int,
    ) -> None:
        """하나의 완료된 LLM 요청에 대한 토큰 수를 누적합니다.

        세션 총계와 모델별 분석을 모두 업데이트합니다.

Args:
            model_name: 이 요청을 처리한 모델입니다(모델별 키로 사용됨). 이 요청에 대한 모델별 분석을 건너뛰려면 빈 문자열을
                        전달하세요.
            input_toks: 이 요청에 대한 토큰을 입력하세요.
            output_toks: 이 요청에 대한 출력 토큰입니다.

        """
        self.request_count += 1
        self.input_tokens += input_toks
        self.output_tokens += output_toks
        if model_name:
            entry = self.per_model.setdefault(model_name, ModelStats())
            entry.request_count += 1
            entry.input_tokens += input_toks
            entry.output_tokens += output_toks

    def merge(self, other: SessionStats) -> None:
        """다른 `SessionStats`을 이 항목에 병합합니다(*self*를 변경함).

        턴별 통계를 세션 수준 합계로 누적하는 데 사용됩니다.

Args:
            other: 접을 수 있는 통계입니다.

        """
        self.request_count += other.request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.wall_time_seconds += other.wall_time_seconds
        for model, ms in other.per_model.items():
            entry = self.per_model.setdefault(model, ModelStats())
            entry.request_count += ms.request_count
            entry.input_tokens += ms.input_tokens
            entry.output_tokens += ms.output_tokens


def format_token_count(count: int) -> str:
    """토큰 수를 사람이 읽을 수 있는 짧은 문자열로 형식화합니다.

Args:
        count: 토큰 수.

Returns:
        `'12.5K'`, `'1.2M'` 또는 `'500'`와 같은 형식화된 문자열입니다.

    """
    if count >= 1_000_000:  # noqa: PLR2004
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:  # noqa: PLR2004
        return f"{count / 1000:.1f}K"
    return str(count)
