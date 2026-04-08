"""플러그형 파일 스토리지를 위한 메모리 백엔드 패키지.

이 패키지는 Deep Agents 프레임워크에서 파일을 저장하고 관리하는 다양한
백엔드 구현체를 제공합니다. 모든 백엔드는 `BackendProtocol`을 구현하여
통일된 파일 작업 인터페이스를 보장합니다.

## 백엔드 계층 구조

```
BackendProtocol (추상 기본 클래스)
├── StateBackend          — 에이전트 상태에 임시 저장 (LangGraph 체크포인팅)
├── StoreBackend          — LangGraph BaseStore에 영구 저장 (크로스 스레드)
├── FilesystemBackend     — 로컬 파일시스템 직접 읽기/쓰기
│   └── LocalShellBackend — 파일시스템 + 로컬 셸 명령 실행
├── CompositeBackend      — 경로 접두사별 라우팅 (하이브리드 구성)
│
└── SandboxBackendProtocol (실행 확장)
    └── BaseSandbox       — execute()를 핵심으로 하는 추상 샌드박스 기반
        └── LangSmithSandbox  — LangSmith 샌드박스 구현
```

## 선택 가이드

| 용도 | 추천 백엔드 | 영구 저장 | 셸 실행 |
|------|-------------|----------|--------|
| 기본/테스트 | StateBackend | X (스레드 내) | X |
| 메모리 영구 보관 | StoreBackend | O (크로스 스레드) | X |
| 로컬 개발 CLI | LocalShellBackend | O (파일시스템) | O |
| 원격 샌드박스 | LangSmithSandbox | O | O |
| 하이브리드 | CompositeBackend | 라우트별 다름 | 기본 백엔드에 의존 |
"""

# === 경로 접두사별 라우팅을 지원하는 복합 백엔드 ===
from deepagents.backends.composite import CompositeBackend

# === 로컬 파일시스템 직접 접근 백엔드 ===
from deepagents.backends.filesystem import FilesystemBackend

# === LangSmith 샌드박스 백엔드 (원격 실행 환경) ===
from deepagents.backends.langsmith import LangSmithSandbox

# === 로컬 셸 실행을 지원하는 파일시스템 백엔드 ===
from deepagents.backends.local_shell import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend

# === 모든 백엔드의 추상 인터페이스 ===
from deepagents.backends.protocol import BackendProtocol

# === LangGraph 에이전트 상태에 임시 저장하는 백엔드 ===
from deepagents.backends.state import StateBackend

# === LangGraph BaseStore에 영구 저장하는 백엔드 ===
from deepagents.backends.store import (
    BackendContext,     # 네임스페이스 팩토리에 전달되는 컨텍스트
    NamespaceFactory,   # 네임스페이스 동적 생성을 위한 타입 별칭
    StoreBackend,       # 영구 저장 백엔드 구현체
)

# 패키지의 공개 API 목록
__all__ = [
    "DEFAULT_EXECUTE_TIMEOUT",   # 셸 명령 기본 타임아웃 (초)
    "BackendContext",            # 네임스페이스 팩토리 컨텍스트
    "BackendProtocol",           # 백엔드 추상 인터페이스
    "CompositeBackend",          # 경로별 라우팅 복합 백엔드
    "FilesystemBackend",         # 로컬 파일시스템 백엔드
    "LangSmithSandbox",          # LangSmith 샌드박스 백엔드
    "LocalShellBackend",         # 로컬 셸 실행 백엔드
    "NamespaceFactory",          # 네임스페이스 팩토리 타입
    "StateBackend",              # 에이전트 상태 임시 백엔드
    "StoreBackend",              # BaseStore 영구 백엔드
]
