"""프로젝트 수준 MCP 서버 구성을 위한 신뢰 저장소입니다.

stdio 서버(로컬 명령 실행)가 포함된 프로젝트 수준 MCP 구성의 지속적인 승인을 관리합니다. 신뢰는 지문 기반입니다. 구성 콘텐츠가 변경되면 사용자가
다시 승인해야 합니다.

신뢰 항목은 `[mcp_trust.projects]` 아래의 `~/.deepagents/config.toml`에 저장됩니다.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path.home() / ".deepagents"
_DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "config.toml"


def compute_config_fingerprint(config_paths: list[Path]) -> str:
    """정렬되고 연결된 구성 콘텐츠에 대해 SHA-256 지문을 계산합니다.

Args:
        config_paths: 지문을 생성할 구성 파일의 경로입니다.

Returns:
        Fingerprint string in the form `sha256: <16진수>`.

    """
    hasher = hashlib.sha256()
    for path in sorted(config_paths):
        try:
            hasher.update(path.read_bytes())
        except OSError:
            logger.warning("Could not read %s for fingerprinting", path, exc_info=True)
    return f"sha256:{hasher.hexdigest()}"


def _load_config(config_path: Path) -> dict[str, Any]:
    """TOML 구성 파일을 읽습니다.

Returns:
        구문 분석된 TOML 데이터 또는 실패 시 빈 사전입니다.

    """
    import tomllib

    try:
        if not config_path.exists():
            return {}
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        logger.debug("Could not read config %s", config_path, exc_info=True)
        return {}


def _save_config(data: dict[str, Any], config_path: Path) -> bool:
    """config_path에 TOML 데이터를 원자적으로 기록합니다.

    충돌 안전을 위해 `tempfile.mkstemp` + `Path.replace`을 사용합니다.

Args:
        data: 전체 TOML 데이터 딕셔너리를 작성합니다.
        config_path: 대상 경로.

Returns:
        성공 시 `True`, I/O 실패 시 `False`.

    """
    import tomli_w

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                tomli_w.dump(data, f)
            Path(tmp_path).replace(config_path)
        except BaseException:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
            raise
    except (OSError, ValueError):
        logger.exception("Failed to save config to %s", config_path)
        return False
    return True


def is_project_mcp_trusted(
    project_root: str,
    fingerprint: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """프로젝트의 MCP 구성이 지정된 지문으로 신뢰되는지 확인하세요.

Args:
        project_root: 프로젝트 루트의 절대 경로입니다.
        fingerprint: 예상되는 지문 문자열(`sha256:<hex>`)입니다.
        config_path: 신뢰 구성 파일의 경로입니다.

Returns:
        `True` 저장된 지문이 일치하는 경우.

    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    projects = data.get("mcp_trust", {}).get("projects", {})
    return projects.get(project_root) == fingerprint


def trust_project_mcp(
    project_root: str,
    fingerprint: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """프로젝트의 MCP 구성에 대한 신뢰를 유지합니다.

Args:
        project_root: 프로젝트 루트의 절대 경로입니다.
        fingerprint: 저장할 지문(`sha256:<hex>`).
        config_path: 신뢰 구성 파일의 경로입니다.

Returns:
        `True` 항목이 성공적으로 저장된 경우.

    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    if "mcp_trust" not in data:
        data["mcp_trust"] = {}
    if "projects" not in data["mcp_trust"]:
        data["mcp_trust"]["projects"] = {}
    data["mcp_trust"]["projects"][project_root] = fingerprint
    return _save_config(data, config_path)


def revoke_project_mcp_trust(
    project_root: str,
    *,
    config_path: Path | None = None,
) -> bool:
    """프로젝트의 MCP 구성에 대한 신뢰를 제거합니다.

Args:
        project_root: 프로젝트 루트의 절대 경로입니다.
        config_path: 신뢰 구성 파일의 경로입니다.

Returns:
        `True` 항목이 제거된 경우(또는 존재하지 않는 경우)

    """
    if config_path is None:
        config_path = _DEFAULT_CONFIG_PATH

    data = _load_config(config_path)
    projects = data.get("mcp_trust", {}).get("projects", {})
    if project_root not in projects:
        return True
    del data["mcp_trust"]["projects"][project_root]
    return _save_config(data, config_path)
