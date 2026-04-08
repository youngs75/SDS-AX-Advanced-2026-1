"""사기성 텍스트 및 URL 검사를 위한 유니코드 보안 도우미입니다.

이 모듈은 의도적으로 경량이므로 시작 성능에 영향을 주지 않고 표시 및 승인 경로로 가져올 수 있습니다.
"""

from __future__ import annotations

import ipaddress
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

_DANGEROUS_CODEPOINTS: frozenset[int] = frozenset(
    {
        # BiDi directional formatting controls (embeddings, overrides, pop)
        *range(0x202A, 0x202F),
        # BiDi isolate controls (isolates, pop isolate)
        *range(0x2066, 0x206A),
        # Zero-width and invisible formatting controls
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        0x2060,  # WORD JOINER
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        # Other commonly abused invisible controls
        0x00AD,  # SOFT HYPHEN
        0x034F,  # COMBINING GRAPHEME JOINER
        0x115F,  # HANGUL CHOSEONG FILLER
        0x1160,  # HANGUL JUNGSEONG FILLER
    }
)
"""CLI 안전을 위해 사기성/보이지 않는 것으로 처리되어야 하는 코드 포인트입니다."""

_DANGEROUS_CHARACTERS: frozenset[str] = frozenset(
    chr(codepoint) for codepoint in _DANGEROUS_CODEPOINTS
)

# Minimal high-risk confusables for warn-level detection.
CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "\u0430": "a",  # CYRILLIC SMALL LETTER A
    "\u0435": "e",  # CYRILLIC SMALL LETTER IE
    "\u043e": "o",  # CYRILLIC SMALL LETTER O
    "\u0440": "p",  # CYRILLIC SMALL LETTER ER
    "\u0441": "c",  # CYRILLIC SMALL LETTER ES
    "\u0443": "y",  # CYRILLIC SMALL LETTER U
    "\u0445": "x",  # CYRILLIC SMALL LETTER HA
    "\u043d": "h",  # CYRILLIC SMALL LETTER EN
    "\u0456": "i",  # CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I
    "\u0458": "j",  # CYRILLIC SMALL LETTER JE
    "\u043a": "k",  # CYRILLIC SMALL LETTER KA
    "\u0455": "s",  # CYRILLIC SMALL LETTER DZE
    # Greek
    "\u03b1": "a",  # GREEK SMALL LETTER ALPHA
    "\u03b5": "e",  # GREEK SMALL LETTER EPSILON
    "\u03bf": "o",  # GREEK SMALL LETTER OMICRON
    "\u03c1": "p",  # GREEK SMALL LETTER RHO
    "\u03c7": "x",  # GREEK SMALL LETTER CHI
    "\u03ba": "k",  # GREEK SMALL LETTER KAPPA
    "\u03bd": "v",  # GREEK SMALL LETTER NU
    "\u03c4": "t",  # GREEK SMALL LETTER TAU
    # Armenian
    "\u0570": "h",  # ARMENIAN SMALL LETTER HO
    "\u0578": "n",  # ARMENIAN SMALL LETTER VO
    "\u057d": "u",  # ARMENIAN SMALL LETTER SEH
    # Fullwidth Latin
    "\uff41": "a",  # FULLWIDTH LATIN SMALL LETTER A
    "\uff45": "e",  # FULLWIDTH LATIN SMALL LETTER E
    "\uff4f": "o",  # FULLWIDTH LATIN SMALL LETTER O
}

URL_ARG_KEYS: frozenset[str] = frozenset(
    {"url", "uri", "href", "link", "base_url", "endpoint"}
)
"""URL을 포함할 가능성이 높으며 안전 확인이 필요한 인수 키 이름입니다."""

_URL_SAFE_LOCAL_HOSTS: frozenset[str] = frozenset({"localhost"})


@dataclass(frozen=True, slots=True)
class UnicodeIssue:
    """텍스트에서 위험한 유니코드 문자가 발견되었습니다.

Attributes:
        position: 원래 문자열의 0부터 시작하는 인덱스입니다.
        character: 입력에서 발견된 단일 원시 문자입니다.
        codepoint: ``U+202E``과 같은 대문자 코드 포인트 문자열입니다.
        name: 유니코드 문자 이름입니다.

    """

    position: int
    character: str
    codepoint: str
    name: str

    def __post_init__(self) -> None:  # noqa: D105
        if len(self.character) != 1:
            msg = (
                "character must be a single code point, "
                f"got length {len(self.character)}"
            )
            raise ValueError(msg)
        expected = f"U+{ord(self.character):04X}"
        if self.codepoint != expected:
            msg = (
                f"codepoint {self.codepoint!r} does not match "
                f"character (expected {expected})"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class UrlSafetyResult:
    """URL 문자열에 대한 안전 분석 출력입니다.

    의심스러운 패턴 없이 정보 경고(예: 퓨니코드 디코딩)가 있는 경우 결과에는 비어 있지 않은 `warnings`이 있는 `safe=True`이 있을 수
    있습니다.

Attributes:
        safe: `True` 의심스러운 패턴이 발견되지 않은 경우.
        decoded_domain: Punycode로 디코딩된 호스트 이름이 원래 호스트 이름과 다른 경우.

            `None`은 변경되지 않았거나 호스트 이름이 존재하지 않는 경우입니다.
        warnings: 사람이 읽을 수 있는 경고 문자열(불변)
        issues: 전체 URL에서 위험한 유니코드 문제가 발견되었습니다(불변).

    """

    safe: bool
    decoded_domain: str | None
    warnings: tuple[str, ...]
    issues: tuple[UnicodeIssue, ...]


def detect_dangerous_unicode(text: str) -> list[UnicodeIssue]:
    """텍스트에서 사기성 또는 숨겨진 유니코드 코드 포인트를 탐지합니다.

Args:
        text: 검사할 텍스트를 입력하세요.

Returns:
        소스 순서대로 `UnicodeIssue` 항목 목록입니다.

    """
    issues: list[UnicodeIssue] = []
    for position, character in enumerate(text):
        if character not in _DANGEROUS_CHARACTERS:
            continue
        issues.append(
            UnicodeIssue(
                position=position,
                character=character,
                codepoint=_format_codepoint(character),
                name=_unicode_name(character),
            )
        )
    return issues


def strip_dangerous_unicode(text: str) -> str:
    """텍스트에서 알려진 위험하거나 보이지 않는 유니코드 문자를 제거합니다.

Args:
        text: 정리할 텍스트를 입력하세요.

Returns:
        위험한 문자가 제거된 텍스트를 정리했습니다.

    """
    return "".join(ch for ch in text if ch not in _DANGEROUS_CHARACTERS)


def render_with_unicode_markers(text: str) -> str:
    """숨겨진 유니코드 문자를 명시적 마커로 렌더링합니다.

    출력 예: `abc<U+202E RIGHT-TO-LEFT OVERRIDE>def`.

Args:
        text: 렌더링할 텍스트를 입력합니다.

Returns:
        위험한 문자가 보이는 마커로 대체되는 텍스트입니다.

    """
    rendered_parts: list[str] = []
    for character in text:
        if character not in _DANGEROUS_CHARACTERS:
            rendered_parts.append(character)
            continue
        rendered_parts.append(
            f"<{_format_codepoint(character)} {_unicode_name(character)}>"
        )
    return "".join(rendered_parts)


def summarize_issues(issues: list[UnicodeIssue], *, max_items: int = 3) -> str:
    """경고 메시지에 대한 유니코드 문제를 요약합니다.

    코드 포인트별로 중복을 제거합니다. *max_items*개 이상의 고유 항목이 존재하는 경우 요약은 `+N more entries` 접미사로 잘립니다.

Args:
        issues: 감지된 문제 목록입니다.
        max_items: 출력에 포함할 최대 고유 코드 포인트입니다.

Returns:
        쉼표로 구분된 요약(예:
            `U+202E RIGHT-TO-LEFT OVERRIDE, U+200B ZERO WIDTH SPACE`.

    """
    unique_entries: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        entry = f"{issue.codepoint} {issue.name}"
        if entry in seen:
            continue
        seen.add(entry)
        unique_entries.append(entry)

    if len(unique_entries) <= max_items:
        return ", ".join(unique_entries)

    displayed = ", ".join(unique_entries[:max_items])
    remainder = len(unique_entries) - max_items
    suffix = "entry" if remainder == 1 else "entries"
    return f"{displayed}, +{remainder} more {suffix}"


def format_warning_detail(warnings: tuple[str, ...], *, max_shown: int = 2) -> str:
    """오버플로 표시기를 사용하여 안전 경고를 표시 문자열에 결합합니다.

Args:
        warnings: `UrlSafetyResult`의 경고 문자열입니다.
        max_shown: 자르기 전에 포함할 최대 경고입니다.

Returns:
        세미콜론으로 구분된 세부 문자열(예: `'warn1; warn2; +1 more'`.

    """
    shown = warnings[:max_shown]
    detail = "; ".join(shown)
    remaining = len(warnings) - max_shown
    if remaining > 0:
        detail += f"; +{remaining} more"
    return detail


def check_url_safety(url: str) -> UrlSafetyResult:
    """의심스러운 유니코드 및 도메인 스푸핑 패턴이 있는지 URL을 확인하세요.

Args:
        url: 검사할 URL 문자열입니다.

Returns:
        `UrlSafetyResult` 디코딩된 도메인 및 경고 세부정보를 포함합니다.

    """
    warnings: list[str] = []
    suspicious = False

    issues = detect_dangerous_unicode(url)
    if issues:
        suspicious = True
        warnings.append(
            f"URL contains hidden Unicode characters ({summarize_issues(issues)})"
        )

    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        return UrlSafetyResult(
            safe=not suspicious,
            decoded_domain=None,
            warnings=tuple(warnings),
            issues=tuple(issues),
        )

    decoded_hostname, failed_punycode = _decode_hostname(hostname)
    decoded_domain = decoded_hostname if decoded_hostname != hostname else None
    if decoded_domain:
        warnings.append(f"Punycode domain decodes to '{decoded_domain}'")
    if failed_punycode:
        suspicious = True
        labels = ", ".join(failed_punycode)
        warnings.append(f"Punycode label(s) could not be decoded: {labels}")

    if _is_local_or_ip_hostname(decoded_hostname):
        return UrlSafetyResult(
            safe=not suspicious,
            decoded_domain=decoded_domain,
            warnings=tuple(warnings),
            issues=tuple(issues),
        )

    for label in _split_hostname_labels(decoded_hostname):
        scripts = _scripts_in_label(label)
        if len(scripts) > 1:
            suspicious = True
            script_names = ", ".join(sorted(scripts))
            warnings.append(f"Domain label '{label}' mixes scripts ({script_names})")

        if _label_has_suspicious_confusable_mix(label):
            suspicious = True
            warnings.append(
                f"Domain label '{label}' contains confusable Unicode characters"
            )

    return UrlSafetyResult(
        safe=not suspicious,
        decoded_domain=decoded_domain,
        warnings=tuple(warnings),
        issues=tuple(issues),
    )


def _decode_hostname(hostname: str) -> tuple[str, list[str]]:
    """가능하면 `xn--` 퓨니코드 레이블을 유니코드 레이블로 디코딩합니다.

Returns:
        (디코딩된 호스트 이름, 디코딩에 실패한 라벨 목록)의 튜플입니다.

    """
    decoded_labels: list[str] = []
    failed_labels: list[str] = []
    for label in _split_hostname_labels(hostname):
        if label.startswith("xn--"):
            try:
                decoded_labels.append(label.encode("ascii").decode("idna"))
            except UnicodeError:
                decoded_labels.append(label)
                failed_labels.append(label)
            continue
        decoded_labels.append(label)
    return ".".join(decoded_labels), failed_labels


def _split_hostname_labels(hostname: str) -> list[str]:
    """호스트 이름을 비어 있지 않은 라벨로 분할합니다.

Returns:
        빈 항목이 없는 호스트 이름 라벨입니다.

    """
    return [label for label in hostname.split(".") if label]


def _is_local_or_ip_hostname(hostname: str) -> bool:
    """호스트 이름이 localhost인지 아니면 IP 주소 리터럴인지 반환합니다.

Returns:
        호스트 이름이 localhost 또는 IP 리터럴인 경우 `True`, 그렇지 않은 경우 `False`.

    """
    host = hostname.strip().rstrip(".")
    if not host:
        return False

    if host.lower() in _URL_SAFE_LOCAL_HOSTS:
        return True

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _scripts_in_label(label: str) -> set[str]:
    """도메인 라벨에서 사용되는 일반적이지 않은 스크립트를 수집합니다.

Returns:
        공통/상속을 제외하고 레이블에서 사용하는 스크립트 이름 집합입니다.

    """
    scripts: set[str] = set()
    for character in label:
        script = _char_script(character)
        if script in {"Common", "Inherited"}:
            continue
        scripts.add(script)
    return scripts


def _label_has_suspicious_confusable_mix(label: str) -> bool:
    """레이블에 사기성 혼동 가능성이 있는 문자가 있는지 여부를 반환합니다.

    혼란스러운 문자를 포함하면서 여러 스크립트를 혼합하는 레이블에만 플래그를 지정합니다. 단일 스크립트 레이블(혼란 가능한 레이블 포함)은 해당 스크립트의
    합법적인 사용을 나타내기 때문에 플래그가 지정되지 않습니다.

Returns:
        `True` 라벨에 스크립트가 혼합되어 있고 혼동하기 쉬운 문자가 포함되어 있는 경우.

    """
    if not any(character in CONFUSABLES for character in label):
        return False

    scripts = _scripts_in_label(label)
    return len(scripts) > 1


def _char_script(character: str) -> str:
    """문자를 거친 유니코드 스크립트 버킷으로 분류합니다.

Returns:
        One of: `'Fullwidth'`, `'Latin'`, `'Cyrillic'`, `'Greek'`, `'Armenian'`,
                `'EastAsian'`, `'Inherited'`, `'Common'` 또는 `'Other'`.

    """
    name = unicodedata.name(character, "")
    category = unicodedata.category(character)

    if "FULLWIDTH LATIN" in name:
        return "Fullwidth"
    if "LATIN" in name:
        return "Latin"
    if "CYRILLIC" in name:
        return "Cyrillic"
    if "GREEK" in name:
        return "Greek"
    if "ARMENIAN" in name:
        return "Armenian"
    if any(
        token in name
        for token in (
            "CJK",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "BOPOMOFO",
            "IDEOGRAPHIC",
        )
    ):
        return "EastAsian"

    if category.startswith("M"):
        return "Inherited"
    if category[0] in {"N", "P", "S", "Z", "C"}:
        return "Common"

    return "Other"


def _format_codepoint(character: str) -> str:
    """문자 코드 포인트를 `U+XXXX` 대문자 형식으로 지정합니다.

Returns:
        대문자 `U+XXXX` 코드포인트 문자열.

    """
    return f"U+{ord(character):04X}"


def _unicode_name(character: str) -> str:
    """알 수 없는 코드 포인트를 대체하여 안정적인 유니코드 이름을 반환합니다.

Returns:
        문자의 유니코드 이름 문자열입니다.

    """
    return unicodedata.name(character, "UNKNOWN CHARACTER")


# ---------------------------------------------------------------------------
# Shared helpers for recursive argument inspection
# ---------------------------------------------------------------------------


def iter_string_values(
    data: dict[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, str]]:
    """중첩된 사전/목록 구조를 키-경로/문자열 쌍으로 평면화합니다.

Returns:
        모든 문자열 리프에 대한 ``(path, value)`` 튜플 목록입니다.

    """
    values: list[tuple[str, str]] = []
    for key, value in data.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, str):
            values.append((key_path, value))
            continue
        if isinstance(value, dict):
            values.extend(iter_string_values(value, prefix=key_path))
            continue
        if isinstance(value, list):
            values.extend(_iter_string_values_from_list(value, prefix=key_path))
    return values


def _iter_string_values_from_list(
    values: list[Any],
    *,
    prefix: str,
) -> list[tuple[str, str]]:
    """중첩된 목록 값을 키-경로/문자열 쌍으로 평면화합니다.

Returns:
        모든 문자열 리프에 대한 `(path, value)` 튜플 목록입니다.

    """
    entries: list[tuple[str, str]] = []
    for index, value in enumerate(values):
        key_path = f"{prefix}[{index}]"
        if isinstance(value, str):
            entries.append((key_path, value))
            continue
        if isinstance(value, dict):
            entries.extend(iter_string_values(value, prefix=key_path))
            continue
        if isinstance(value, list):
            entries.extend(_iter_string_values_from_list(value, prefix=key_path))
    return entries


def looks_like_url_key(arg_path: str) -> bool:
    """키 경로가 URL과 유사한 콘텐츠를 제안하는지 여부를 반환합니다.

Returns:
        URL과 유사한 키 이름의 경우 `True`, 그렇지 않은 경우 `False`.

    """
    key = arg_path.rsplit(".", maxsplit=1)[-1]
    key = key.split("[", maxsplit=1)[0].lower()
    return key in URL_ARG_KEYS
