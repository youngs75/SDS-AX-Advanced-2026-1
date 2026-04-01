# Function Calling vs Structured Output

## 학습 목표

- FC와 SO의 실제 성능 차이 측정 (레이턴시, 토큰 비용)
- LLM 관점에서 FC와 SO의 본질적 동일성 이해
- SLM/불안정 모델 대응: SO Fallback 패턴 구현
- 실무 의사결정 가이드 자동 생성

## 핵심 인사이트

**LLM 입장에서는 결국 모든 것이 Structured Output입니다.**

Function Calling은 사용자 편의를 위한 Message wrapper일 뿐, 내부적으로는 JSON Schema를 기반으로 한 Structured Output으로 변환됩니다.
