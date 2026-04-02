"""DeepEval 전용 OpenRouter 모델 래퍼.

DeepEval 라이브러리의 Synthesizer, Metric 등은 model= 인자로
DeepEvalBaseLLM 인스턴스를 받습니다. 이 모듈은 OpenRouter를
DeepEval이 이해할 수 있는 형태로 감싸는 어댑터 패턴을 구현합니다.

아키텍처:
    OpenRouter API ← OpenAI SDK ← OpenRouterModel(DeepEvalBaseLLM)
                                   ↑
                      DeepEval Synthesizer/Metric이 이 모델을 호출

사용 예시:
    from src.llm.deepeval_model import get_deepeval_model
    model = get_deepeval_model()

    # Synthesizer에 전달
    synthesizer = Synthesizer(model=model)

    # Metric에 전달
    metric = AnswerRelevancyMetric(model=model, threshold=0.7)
"""

from __future__ import annotations

import json

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

from src.settings import get_settings


class OpenRouterModel(DeepEvalBaseLLM):
    """DeepEval 전용 OpenRouter 모델 래퍼.

    DeepEvalBaseLLM을 상속하여 DeepEval 내부에서 사용하는
    generate(), a_generate(), get_model_name() 인터페이스를 구현합니다.

    내부적으로 OpenAI SDK + OpenRouter base_url을 사용하여 LLM을 호출합니다.

    Args:
        model_name: 사용할 모델명 (예: "openai/gpt-4.1"). None이면 settings에서 읽음
        api_key: OpenRouter API 키. None이면 settings에서 읽음
    """

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.openrouter_model_name
        # OpenAI SDK를 OpenRouter 엔드포인트로 초기화
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or settings.openrouter_api_key,
        )

    def load_model(self):
        """모델 클라이언트를 반환합니다 (DeepEvalBaseLLM 인터페이스)."""
        return self._client

    def generate(self, prompt: str, schema=None):
        """동기 텍스트 생성.

        DeepEval이 메트릭 평가 시 호출하는 핵심 메서드입니다.
        temperature=0.0으로 결정론적 출력을 보장합니다.

        Args:
            prompt: LLM에 전달할 프롬프트 텍스트
            schema: (선택) Pydantic 모델. 지정 시 JSON 파싱 후 스키마 검증

        Returns:
            schema 미지정 시 LLM 응답 문자열.
            schema 지정 시 파싱 성공하면 schema 인스턴스, 실패하면 원문 문자열.
        """
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""

        # schema가 지정된 경우: JSON 파싱 → Pydantic 검증
        if schema:
            try:
                parsed = json.loads(content)
                return schema(**parsed)
            except Exception as exc:
                print(f"[WARN] Schema JSON parse failed ({schema.__name__}): {exc}")
                try:
                    return schema(response=content)
                except Exception:
                    return content
        return content

    async def a_generate(self, prompt: str, schema=None):
        """비동기 텍스트 생성 (동기 generate를 위임).

        DeepEval의 비동기 평가 파이프라인에서 호출됩니다.
        schema 지정 시 generate()와 동일하게 schema 인스턴스를 반환할 수 있습니다.
        """
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        """모델 식별자를 반환합니다 (예: "openai/gpt-4.1")."""
        return self._model_name


def get_deepeval_model(model_name: str | None = None) -> OpenRouterModel:
    """DeepEval 메트릭/합성기에 전달할 OpenRouter 모델 인스턴스를 생성합니다.

    팩토리 함수로, 매 호출마다 새 인스턴스를 생성합니다.
    (메트릭마다 독립적인 모델 인스턴스가 필요할 수 있으므로)

    Args:
        model_name: 사용할 모델명. None이면 settings의 기본값 사용

    Returns:
        OpenRouterModel: DeepEval 호환 모델 인스턴스
    """
    return OpenRouterModel(model_name=model_name)
