"""`deepagents-cli`의 수명 주기를 업데이트합니다.

PyPI에 대한 버전 확인(캐싱 포함), 설치 방법 감지, 자동 업그레이드 실행, 구성 기반 옵트인/아웃 및 "새로운 기능" 추적을 처리합니다.

대부분의 공개 진입점은 오류를 흡수하고 감시 값을 반환합니다. `set_auto_update`은 쓰기 실패 시 발생하므로 호출자가 조치 가능한 피드백을 표시할
수 있습니다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import tomllib
from typing import TYPE_CHECKING, Literal

from packaging.version import InvalidVersion, Version

from deepagents_cli._version import PYPI_URL, USER_AGENT, __version__

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_cli.model_config import DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

CACHE_FILE: Path = DEFAULT_CONFIG_DIR / "latest_version.json"
SEEN_VERSION_FILE: Path = DEFAULT_CONFIG_DIR / "seen_version.json"
CACHE_TTL = 86_400  # 24 hours

InstallMethod = Literal["uv", "pip", "brew", "unknown"]

_UPGRADE_COMMANDS: dict[InstallMethod, str] = {
    "uv": "uv tool upgrade deepagents-cli",
    "brew": "brew upgrade deepagents-cli",
    "pip": "pip install --upgrade deepagents-cli",
}
"""설치 방법에 따른 업그레이드 명령입니다.

`perform_upgrade`은 감지된 설치 방법과 일치하는 명령만 실행합니다. 대체 체인이 없습니다.
"""

_UPGRADE_TIMEOUT = 120  # seconds


def _parse_version(v: str) -> Version:
    """PEP 440 버전 문자열을 비교 가능한 `Version` 객체로 구문 분석합니다.

    안정 버전(`1.2.3`) 및 시험판(`1.2.3a1`, `1.2.3rc2`) 버전을 지원합니다.

Args:
        v: `'1.2.3'` 또는 `'1.2.3a1'`과 같은 버전 문자열입니다.

Returns:
        `packaging.version.Version` 인스턴스.

    """
    return Version(v.strip())  # raises InvalidVersion for non-PEP 440 strings


def _latest_from_releases(
    releases: dict[str, list[object]],
    *,
    include_prereleases: bool,
) -> str | None:
    """PyPI `releases` 매핑에서 최신 버전을 선택하세요.

    업로드된 파일이 없는 버전(빈 항목)을 건너뛰고, *include_prereleases*가 `False`인 경우 시험판 버전을 건너뜁니다.

Args:
        releases: PyPI JSON API의 `releases` dict.
        include_prereleases: 시험판 버전을 고려할지 여부입니다.

Returns:
        가장 일치하는 버전 문자열이거나, 해당하는 문자열이 없으면 `None`입니다.

    """
    best: Version | None = None
    best_str: str | None = None
    for ver_str, files in releases.items():
        if not files:
            continue
        try:
            ver = Version(ver_str)
        except InvalidVersion:
            logger.debug("Skipping unparseable release key: %s", ver_str)
            continue
        if not include_prereleases and ver.is_prerelease:
            continue
        if best is None or ver > best:
            best = ver
            best_str = ver_str
    return best_str


def get_latest_version(
    *,
    bypass_cache: bool = False,
    include_prereleases: bool = False,
) -> str | None:
    """캐싱을 사용하여 PyPI에서 최신 deepagents-cli 버전을 가져옵니다.

    반복적인 네트워크 호출을 피하기 위해 결과는 `CACHE_FILE`에 캐시됩니다. 캐시는 최신 안정 버전과 시험판 버전을 모두 저장하므로 단일 PyPI
    요청이 두 코드 경로를 모두 제공합니다.

Args:
        bypass_cache: 캐시를 건너뛰고 항상 PyPI를 실행하세요.
        include_prereleases: `True`인 경우 시험판 버전(알파, 베타, rc)을 고려하세요. 안정적인 사용자는 이 `False`을
                             떠나야 합니다.

Returns:
        최신 버전 문자열 또는 실패 시 `None`.

    """
    cache_key = "version_prerelease" if include_prereleases else "version"

    try:
        if not bypass_cache and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            fresh = time.time() - data.get("checked_at", 0) < CACHE_TTL
            if fresh and cache_key in data:
                return data[cache_key]
    except (OSError, json.JSONDecodeError, TypeError):
        logger.debug("Failed to read update-check cache", exc_info=True)

    try:
        import requests
    except ImportError:
        logger.warning(
            "requests package not installed — update checks disabled. "
            "Install with: pip install requests"
        )
        return None

    try:
        resp = requests.get(
            PYPI_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=3,
        )
        resp.raise_for_status()
        payload = resp.json()
        stable: str = payload["info"]["version"]
        releases: dict[str, list[object]] = payload.get("releases", {})
        if not releases:
            logger.debug("PyPI response missing or empty 'releases' key")
        prerelease = _latest_from_releases(releases, include_prereleases=True)
    except (requests.RequestException, OSError, KeyError, json.JSONDecodeError):
        logger.debug("Failed to fetch latest version from PyPI", exc_info=True)
        return None

    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps(
                {
                    "version": stable,
                    "version_prerelease": prerelease,
                    "checked_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to write update-check cache", exc_info=True)

    return prerelease if include_prereleases else stable


def is_update_available(*, bypass_cache: bool = False) -> tuple[bool, str | None]:
    """deepagents-cli의 최신 버전을 사용할 수 있는지 확인하세요.

    설치된 버전이 시험판(예: `0.0.35a1`)인 경우 PyPI의 시험판 버전이 비교에 포함되므로 알파 테스터에게 최신 알파와 최종 안정 릴리스에 대한
    알림이 제공됩니다. 안정적인 설치는 안정적인 PyPI 릴리스와만 비교됩니다.

Args:
        bypass_cache: 캐시를 건너뛰고 항상 PyPI를 실행하세요.

Returns:
        `(available, latest)` 튜플.

            PyPI 버전이 설치된 버전보다 엄격히 최신인 경우 `available`은 `True`입니다. `latest`는 버전 문자열입니다(또는
            확인에 실패한 경우 `None`).

    """
    try:
        installed = _parse_version(__version__)
    except InvalidVersion:
        logger.warning(
            "Installed version %r is not PEP 440 compliant; "
            "update checks disabled for this install",
            __version__,
        )
        return False, None

    include_prereleases = installed.is_prerelease
    latest = get_latest_version(
        bypass_cache=bypass_cache,
        include_prereleases=include_prereleases,
    )
    if latest is None:
        return False, None

    try:
        if _parse_version(latest) > installed:
            return True, latest
    except InvalidVersion:
        logger.debug("Failed to compare versions", exc_info=True)

    return False, None


# ---------------------------------------------------------------------------
# Install method detection
# ---------------------------------------------------------------------------


def detect_install_method() -> InstallMethod:
    """`deepagents-cli`이 어떻게 설치되었는지 감지합니다.

    uv 및 Homebrew의 알려진 경로와 비교하여 `sys.prefix`을 확인합니다.

Returns:
        The detected install method: `'uv'`, `'brew'`, `'pip'` 또는 `'unknown'`(편집 가능/개발자
                                     설치).

    """
    from deepagents_cli.config import _is_editable_install

    prefix = sys.prefix
    # uv tool installs live under ~/.local/share/uv/tools/
    if "/uv/tools/" in prefix or "\\uv\\tools\\" in prefix:
        return "uv"
    # Homebrew prefixes
    if any(
        prefix.startswith(p)
        for p in ("/opt/homebrew", "/usr/local/Cellar", "/home/linuxbrew")
    ):
        return "brew"
    # Editable / dev installs — don't auto-upgrade
    if _is_editable_install():
        return "unknown"
    return "pip"


def upgrade_command(method: InstallMethod | None = None) -> str:
    """`deepagents-cli`을(를) 업그레이드하려면 쉘 명령을 반환하십시오.

    인식할 수 없는 설치 방법에 대해서는 pip 명령으로 대체합니다.

Args:
        method: 설치 방법 재정의.

            `None`인 경우 자동 감지됩니다.

    """
    if method is None:
        method = detect_install_method()
    return _UPGRADE_COMMANDS.get(method, _UPGRADE_COMMANDS["pip"])


async def perform_upgrade() -> tuple[bool, str]:
    """감지된 설치 방법을 사용하여 `deepagents-cli` 업그레이드를 시도합니다.

    감지된 방법만 시도합니다. 환경 간 오염을 피하기 위해 다른 패키지 관리자에게 의존하지 않습니다.

Returns:
        `(success, output)` — *출력*은 결합된 stdout/stderr입니다.

    """
    method = detect_install_method()
    if method == "unknown":
        return False, "Editable install detected — skipping auto-update."

    cmd = _UPGRADE_COMMANDS.get(method)
    if cmd is None:
        return False, f"No upgrade command for install method: {method}"

    # Skip brew if binary not on PATH
    if method == "brew" and not shutil.which("brew"):
        return False, "brew not found on PATH."

    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_UPGRADE_TIMEOUT
        )
        output = (stdout or b"").decode() + (stderr or b"").decode()
        if proc.returncode == 0:
            return True, output.strip()
        logger.warning(
            "Upgrade via %s exited with code %d: %s",
            method,
            proc.returncode,
            output.strip(),
        )
        return False, output.strip()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        msg = f"Upgrade command timed out after {_UPGRADE_TIMEOUT}s: {cmd}"
        logger.warning(msg)
        return False, msg
    except OSError:
        logger.warning("Failed to execute upgrade command: %s", cmd, exc_info=True)
        return False, f"Failed to execute: {cmd}"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def is_update_check_enabled() -> bool:
    """업데이트 확인이 활성화되어 있는지 여부를 반환합니다.

    `DEEPAGENTS_CLI_NO_UPDATE_CHECK` env var 및 `config.toml`의 `[update].check` 키를 확인합니다.

    기본값은 활성화입니다.

    """
    from deepagents_cli._env_vars import NO_UPDATE_CHECK

    if os.environ.get(NO_UPDATE_CHECK):
        return False
    return _read_update_config().get("check", True)


def is_auto_update_enabled() -> bool:
    """자동 업데이트 활성화 여부를 반환합니다.

    `DEEPAGENTS_CLI_AUTO_UPDATE=1` env var 또는 `config.toml`의 `[update].auto_update =
    true`을 통해 선택하세요.

    기본값은 `False`입니다.

    편집 가능한 설치의 경우 항상 비활성화됩니다.

    """
    from deepagents_cli._env_vars import AUTO_UPDATE
    from deepagents_cli.config import _is_editable_install

    if _is_editable_install():
        return False
    if os.environ.get(AUTO_UPDATE, "").lower() in {"1", "true", "yes"}:
        return True
    return _read_update_config().get("auto_update", False)


def set_auto_update(enabled: bool) -> None:
    """`config.toml`에 대한 자동 업데이트 기본 설정을 유지합니다.

    설정이 세션 전반에 걸쳐 유지되도록 `[update].auto_update`을 작성합니다.

Args:
        enabled: 자동 업데이트를 활성화해야 하는지 여부입니다.

    """
    import contextlib
    import tempfile
    from pathlib import Path

    import tomli_w

    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    else:
        data = {}

    if "update" not in data:
        data["update"] = {}
    data["update"]["auto_update"] = enabled

    fd, tmp_path = tempfile.mkstemp(dir=DEFAULT_CONFIG_PATH.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            tomli_w.dump(data, f)
        Path(tmp_path).replace(DEFAULT_CONFIG_PATH)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink()
        raise


def _read_update_config() -> dict[str, bool]:
    """`config.toml`에서 `[update]` 섹션을 읽어보세요.

Returns:
        누락되거나 읽을 수 없는 파일에서는 비어 있는 부울 구성 값의 사전입니다.

    """
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            return {}
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
        section = data.get("update", {})
        return {k: v for k, v in section.items() if isinstance(v, bool)}
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read [update] config — using defaults", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# "What's new" tracking
# ---------------------------------------------------------------------------


def get_seen_version() -> str | None:
    """사용자가 "새로운 기능" 배너를 본 마지막 버전을 반환합니다."""
    try:
        if SEEN_VERSION_FILE.exists():
            data = json.loads(SEEN_VERSION_FILE.read_text(encoding="utf-8"))
            return data.get("version")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.debug("Failed to read seen-version file", exc_info=True)
    return None


def mark_version_seen(version: str) -> None:
    """사용자가 *버전*에 대한 "새로운 기능" 배너를 보았다고 기록합니다."""
    try:
        SEEN_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SEEN_VERSION_FILE.write_text(
            json.dumps({"version": version, "seen_at": time.time()}),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("Failed to write seen-version file", exc_info=True)


def should_show_whats_new() -> bool:
    """최신 버전을 처음 실행하는 경우 `True`을 반환합니다."""
    seen = get_seen_version()
    if seen is None:
        # First run ever — mark current as seen, don't show banner.
        mark_version_seen(__version__)
        return False
    try:
        return _parse_version(__version__) > _parse_version(seen)
    except InvalidVersion:
        logger.debug("Failed to compare versions for what's-new check", exc_info=True)
        return False
