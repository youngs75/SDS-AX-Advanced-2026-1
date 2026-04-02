# Langfuse Docs MCP 학습/의사결정 로그 템플릿

아래 템플릿을 복사해 세션별로 기록하세요.

```md
# MCP Study Log - YYYY-MM-DD HH:mm

## 0. Header
- session_id:
- operator:
- objective:
- related_issue:
- target_domain: (observability / prompt-management / evaluation / cross)

## 1. Questions Asked
| ID | Question Template | Why Asked | Priority |
|---|---|---|---|
| O-1 |  |  | High |
| E-1 |  |  | High |
| P-2 |  |  | Medium |

## 2. MCP Query Trail
| Step | MCP Tool | Query / Page | Key Finding | Confidence(1-5) |
|---|---|---|---|---|
| 1 | getLangfuseOverview |  |  |  |
| 2 | searchLangfuseDocs |  |  |  |
| 3 | getLangfuseDocsPage |  |  |  |

## 3. Final Decisions
- decision_1:
  - rationale:
  - expected_impact:
  - risk:
- decision_2:
  - rationale:
  - expected_impact:
  - risk:

## 4. Action Items (Execution-ready)
| Action | Owner | Due | Blocker | Status |
|---|---|---|---|---|
|  |  |  |  | Todo |
|  |  |  |  | Todo |

## 5. Mapping to Day3 Pipeline
- Step5 영향:
- Step6 영향:
- Step8 영향:
- Required code/docs update:

## 6. Verification Plan
- command_1:
- command_2:
- success_criteria:

## 7. Reference Links
- doc_1:
- doc_2:
- doc_3:

## 8. Session Retrospective
- what_worked:
- what_failed:
- next_session_focus:
```

---

## 간단 사용 가이드

1. 질문 전: `Questions Asked` 먼저 채우기
2. 답변 중: `MCP Query Trail`에 근거 페이지 즉시 기록
3. 종료 전: `Final Decisions`와 `Action Items`를 코드 실행 단위로 확정
