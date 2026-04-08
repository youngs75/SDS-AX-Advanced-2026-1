"""이미지/비디오 추적 및 파일 언급 구문 분석을 포함한 입력 처리 유틸리티."""

import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse

from rich.markup import escape as escape_markup

from deepagents_cli.config import console
from deepagents_cli.media_utils import ImageData, VideoData

logger = logging.getLogger(__name__)

PATH_CHAR_CLASS = r"A-Za-z0-9._~/\\:-"
"""파일 경로에 허용되는 문자입니다.

영숫자, 마침표, 밑줄, 물결표(홈), 앞으로/뒤로 슬래시(경로 구분 기호), 콜론(Windows 드라이브 문자) 및 하이픈을 포함합니다.
"""

FILE_MENTION_PATTERN = re.compile(r"@(?P<path>(?:\\.|[" + PATH_CHAR_CLASS + r"])+)")
"""입력 텍스트에서 `@file` 멘션을 추출하는 패턴입니다.

`@` 뒤에 하나 이상의 경로 문자 또는 이스케이프된 문자 쌍(백슬래시 + 모든 문자, 예를 들어 경로의 공백인 경우 `\\ `)과 일치합니다.

경로가 없는 `@`은 유효한 파일 참조가 아니기 때문에 `+`(`*` 아님)을 사용합니다.
"""

EMAIL_PREFIX_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]$")
"""`@` 기호 앞에 오는 이메일과 유사한 텍스트를 감지하는 패턴입니다.

`@` 바로 앞의 문자가 이 패턴과 일치하는 경우 `@mention`은 파일 참조가 아닌 이메일 주소(예: `user@example.com`)의 일부일
가능성이 높습니다.
"""

INPUT_HIGHLIGHT_PATTERN = re.compile(
    r"(^\/[a-zA-Z0-9_-]+|@(?:\\.|[" + PATH_CHAR_CLASS + r"])+)"
)
"""렌더링된 사용자 메시지에서 `@mentions` 및 `/commands`을 강조 표시하는 패턴입니다.

다음 중 하나와 일치합니다. - 문자열 시작 부분의 슬래시 명령(예: `/help`) - `@file`은 텍스트의 어느 위치에서든 언급됩니다(예:
`@README.md`).

참고: `^` 앵커는 줄의 시작이 아니라 문자열의 시작과 일치합니다. `UserMessage.compose()`의 소비자는 슬래시 명령 스타일을 지정하기 전에
`start == 0`을 추가로 확인하므로 `/` 중간 문자열이 강조 표시되지 않습니다.
"""

MediaKind = Literal["image", "video"]
"""`MediaTracker` 메소드의 `kind` 매개변수에 허용되는 값입니다."""

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[image (?P<id>\d+)\]")
"""명명된 `id` 캡처 그룹이 있는 이미지 자리 표시자의 패턴입니다.

추적기가 오래된 항목을 정리하고 사용 가능한 다음 ID를 계산할 수 있도록 자리 표시자 토큰에서 숫자 ID를 추출하는 데 사용됩니다.
"""

VIDEO_PLACEHOLDER_PATTERN = re.compile(r"\[video (?P<id>\d+)\]")
"""명명된 `id` 캡처 그룹이 있는 비디오 자리 표시자의 패턴입니다.

추적기가 오래된 항목을 정리하고 사용 가능한 다음 ID를 계산할 수 있도록 자리 표시자 토큰에서 숫자 ID를 추출하는 데 사용됩니다.
"""

_UNICODE_SPACE_EQUIVALENTS = str.maketrans(
    {
        "\u00a0": " ",  # NO-BREAK SPACE
        "\u202f": " ",  # NARROW NO-BREAK SPACE
    }
)
"""유니코드 공간 변형을 정규화하는 데 사용되는 변환 테이블입니다.

일부 macOS에서 생성된 파일 이름(예: 스크린샷)에는 붙여넣을 때 일반 공백과 동일하게 보이는 비ASCII 공백 코드 포인트가 포함될 수 있습니다.
"""

_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
"""`C:\\Users\\...`와 같은 Windows 드라이브 문자 경로의 패턴입니다."""


@dataclass(frozen=True)
class ParsedPastedPathPayload:
    """삭제된 경로 페이로드 감지에 대한 통합 구문 분석 결과입니다.

Attributes:
        paths: 입력 페이로드에서 구문 분석된 파일 경로가 확인되었습니다.
        token_end: 페이로드가 경로와 후행 텍스트로 시작될 때 구문 분석된 선행 토큰의 끝 인덱스(제외)입니다.

            `None`은 전체 페이로드가 경로 전용 콘텐츠로 구문 분석되었음을 의미합니다.

    """

    paths: list[Path]
    token_end: int | None = None


class MediaTracker:
    """현재 대화에 붙여넣은 이미지와 비디오를 추적합니다."""

    def __init__(self) -> None:
        """빈 미디어 추적기를 초기화합니다.

        이미지와 비디오를 저장하기 위해 빈 목록을 설정하고 고유한 자리 표시자 식별자를 생성하기 위해 ID 카운터를 1로 초기화합니다.

        """
        self.images: list[ImageData] = []
        self.videos: list[VideoData] = []
        self.next_image_id: int = 1
        self.next_video_id: int = 1

    def add_media(self, data: ImageData | VideoData, kind: MediaKind) -> str:
        """미디어 항목을 추가하고 해당 자리 표시자 텍스트를 반환합니다.

Args:
            data: 추적할 이미지 또는 비디오 데이터입니다.
            kind: 미디어 유형 키.

Returns:
            "[image 1]" 또는 "[video 1]"과 같은 자리 표시자 문자열입니다.

        """
        if kind == "image":
            placeholder = f"[image {self.next_image_id}]"
            data.placeholder = placeholder
            self.images.append(data)  # type: ignore[arg-type]
            self.next_image_id += 1
        else:
            placeholder = f"[video {self.next_video_id}]"
            data.placeholder = placeholder
            self.videos.append(data)  # type: ignore[arg-type]
            self.next_video_id += 1
        return placeholder

    def add_image(self, image_data: ImageData) -> str:
        """이미지를 추가하고 자리 표시자 텍스트를 반환합니다.

Args:
            image_data: 추적할 이미지 데이터입니다.

Returns:
            "[image 1]"과 같은 자리 표시자 문자열입니다.

        """
        return self.add_media(image_data, "image")

    def add_video(self, video_data: VideoData) -> str:
        """비디오를 추가하고 자리 표시자 텍스트를 반환합니다.

Args:
            video_data: 추적할 비디오 데이터입니다.

Returns:
            "[동영상 1]"과 같은 자리 표시자 문자열입니다.

        """
        return self.add_media(video_data, "video")

    def get_media(self, kind: MediaKind) -> list[ImageData] | list[VideoData]:
        """특정 유형의 추적된 모든 미디어를 가져옵니다.

Args:
            kind: 미디어 유형 키.

Returns:
            추적된 미디어 항목 목록의 사본입니다.

        """
        if kind == "image":
            return list(self.images)
        return list(self.videos)

    def get_images(self) -> list[ImageData]:
        """추적된 모든 이미지를 가져옵니다.

Returns:
            추적된 이미지 목록의 사본입니다.

        """
        return list(self.images)

    def get_videos(self) -> list[VideoData]:
        """추적된 모든 비디오를 가져옵니다.

Returns:
            추적된 비디오 목록 사본.

        """
        return list(self.videos)

    def clear(self) -> None:
        """추적된 모든 미디어를 지우고 카운터를 재설정합니다."""
        self.images.clear()
        self.videos.clear()
        self.next_image_id = 1
        self.next_video_id = 1

    def sync_to_text(self, text: str) -> None:
        """현재 텍스트의 자리 표시자가 여전히 참조하는 미디어만 유지합니다.

Args:
            text: 사용자에게 표시되는 현재 입력 텍스트입니다.

        """
        img_found = self._sync_kind_images(text)
        vid_found = self._sync_kind_videos(text)
        if not img_found and not vid_found:
            self.clear()

    def _sync_kind_images(self, text: str) -> bool:
        """이미지 목록을 텍스트의 남아 있는 자리 표시자와 동기화합니다.

Args:
            text: 현재 입력 텍스트.

Returns:
            이미지 자리 표시자가 발견되었는지 여부입니다.

        """
        placeholders = {m.group(0) for m in IMAGE_PLACEHOLDER_PATTERN.finditer(text)}
        self.images = [img for img in self.images if img.placeholder in placeholders]
        if not self.images:
            self.next_image_id = 1
        else:
            self.next_image_id = self._max_placeholder_id(
                self.images, IMAGE_PLACEHOLDER_PATTERN, len(self.images)
            )
        return bool(placeholders)

    def _sync_kind_videos(self, text: str) -> bool:
        """비디오 목록을 텍스트의 남은 자리 표시자와 동기화합니다.

Args:
            text: 현재 입력 텍스트.

Returns:
            비디오 자리 표시자가 발견되었는지 여부입니다.

        """
        placeholders = {m.group(0) for m in VIDEO_PLACEHOLDER_PATTERN.finditer(text)}
        self.videos = [vid for vid in self.videos if vid.placeholder in placeholders]
        if not self.videos:
            self.next_video_id = 1
        else:
            self.next_video_id = self._max_placeholder_id(
                self.videos, VIDEO_PLACEHOLDER_PATTERN, len(self.videos)
            )
        return bool(placeholders)

    @staticmethod
    def _max_placeholder_id(
        items: list[ImageData] | list[VideoData],
        pattern: re.Pattern[str],
        fallback_count: int,
    ) -> int:
        """가장 높은 생존 자리 표시자에서 다음 ID를 계산합니다.

Args:
            items: 살아남은 미디어 아이템.
            pattern: `id` 그룹이 있는 자리표시자 정규식입니다.
            fallback_count: 파싱할 수 있는 ID가 없을 때 대체됩니다.

Returns:
            다음 ID 값(max_id + 1).

        """
        max_id = 0
        for item in items:
            match = pattern.fullmatch(item.placeholder)
            if match is not None:
                max_id = max(max_id, int(match.group("id")))
        return max_id + 1 if max_id else fallback_count + 1


def parse_file_mentions(text: str) -> tuple[str, list[Path]]:
    """`@file` 멘션을 추출하고 해결된 파일 경로가 포함된 텍스트를 반환합니다.

    입력 텍스트에서 `@file` 멘션을 구문 분석하고 이를 절대 파일 경로로 확인합니다. 존재하지 않거나 확인할 수 없는 파일은 콘솔에 경고가 인쇄되면서
    제외됩니다.

    이메일 주소(예: `user@example.com`)는 `@` 기호 앞의 이메일 유사 문자를 감지하여 자동으로 제외됩니다.

    경로에서 백슬래시로 이스케이프 처리된 공백(예: `@my\\ folder/file.txt`)은 해결 전에 이스케이프 해제됩니다. 물결표 경로(예:
    `@~/file.txt`)는 `Path.expanduser()`를 통해 확장됩니다. 일반 파일만 반환됩니다. 디렉토리는 제외됩니다.

    이 함수는 예외를 발생시키지 않습니다. 잘못된 경로는 콘솔 경고를 통해 내부적으로 처리됩니다.

Args:
        text: `@file` 멘션이 포함될 가능성이 있는 입력 텍스트입니다.

Returns:
        튜플(원본 텍스트는 변경되지 않고 존재하는 확인된 파일 경로 목록).

    """
    matches = FILE_MENTION_PATTERN.finditer(text)

    files = []
    for match in matches:
        # Skip if this looks like an email address
        text_before = text[: match.start()]
        if text_before and EMAIL_PREFIX_PATTERN.search(text_before):
            continue

        raw_path = match.group("path")
        clean_path = raw_path.replace("\\ ", " ")

        try:
            path = Path(clean_path).expanduser()

            if not path.is_absolute():
                path = Path.cwd() / path

            resolved = path.resolve()
            if resolved.exists() and resolved.is_file():
                files.append(resolved)
            else:
                console.print(
                    f"[yellow]Warning: File not found: "
                    f"{escape_markup(raw_path)}[/yellow]"
                )
        except (OSError, RuntimeError) as e:
            console.print(
                f"[yellow]Warning: Invalid path "
                f"{escape_markup(raw_path)}: "
                f"{escape_markup(str(e))}[/yellow]"
            )

    return text, files


def parse_pasted_file_paths(text: str) -> list[Path]:
    """드래그 앤 드롭된 파일 경로가 포함될 수 있는 붙여넣기 페이로드를 구문 분석합니다.

    파서는 의도적으로 엄격합니다. 전체 붙여넣기 페이로드가 하나 이상의 기존 파일로 해석될 수 있는 경우에만 경로를 반환합니다. 잘못된 토큰은 빈 목록을
    반환하여 일반 텍스트 붙여넣기 동작으로 대체됩니다.

    일반적인 삭제 경로 형식을 지원합니다.

    - 절대/상대 경로 - POSIX 쉘 인용 및 이스케이프 - `file://` URL

Args:
        text: 터미널의 원시 페이스트 페이로드입니다.

Returns:
        확인된 파일 경로 목록 또는 구문 분석이 실패할 경우 빈 목록입니다.

    """
    payload = text.strip()
    if not payload:
        return []

    tokens: list[str] = []
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_tokens = _split_paste_line(line)
        if not line_tokens:
            return []
        tokens.extend(line_tokens)

    if not tokens:
        return []

    paths: list[Path] = []
    for token in tokens:
        path = _token_to_path(token)
        if path is None:
            return []
        resolved = _resolve_existing_pasted_path(path)
        if resolved is None:
            return []
        paths.append(resolved)

    return paths


def parse_pasted_path_payload(
    text: str, *, allow_leading_path: bool = False
) -> ParsedPastedPathPayload | None:
    """하나의 진입점을 통해 삭제된 경로 페이로드 변형을 구문 분석합니다.

    구문 분석 순서는 다음과 같습니다. 1. 엄격한 다중 경로 페이로드 구문 분석(`parse_pasted_file_paths`) 2. 단일 경로
    정규화/파싱(`parse_single_pasted_file_path`) 3. 선택적 선행 경로
    추출(`extract_leading_pasted_file_path`)

Args:
        text: 구문 분석할 입력 페이로드입니다.
        allow_leading_path: 선행 경로 토큰과 후행 프롬프트 텍스트를 구문 분석할지 여부입니다.

Returns:
        파싱된 페이로드 세부정보, 그렇지 않으면 `None`.

    """
    paths = parse_pasted_file_paths(text)
    if paths:
        return ParsedPastedPathPayload(paths=paths)

    single_path = parse_single_pasted_file_path(text)
    if single_path is not None:
        return ParsedPastedPathPayload(paths=[single_path])

    if not allow_leading_path:
        return None

    leading = extract_leading_pasted_file_path(text)
    if leading is None:
        return None

    path, token_end = leading
    return ParsedPastedPathPayload(paths=[path], token_end=token_end)


def parse_single_pasted_file_path(text: str) -> Path | None:
    """붙여넣은 단일 경로 페이로드를 구문 분석하고 해결합니다.

    `parse_pasted_file_paths`과 달리 이 도우미는 하나의 경로 토큰만 허용하며 붙여넣기 이벤트가 단일 경로 표현을 전달하는 경우 대체
    처리를 위한 것입니다.

Args:
        text: 원시 붙여넣은 텍스트 페이로드입니다.

Returns:
        페이로드가 단일 기존 파일인 경우 확인된 경로이고, 그렇지 않은 경우 `None`입니다.

    """
    candidate = normalize_pasted_path(text)
    if candidate is None:
        return None
    return _resolve_existing_pasted_path(candidate)


def extract_leading_pasted_file_path(text: str) -> tuple[Path, int] | None:
    """입력 텍스트에서 붙여넣은 선행 경로 토큰을 추출하고 확인합니다.

    이는 사용자 메시지가 경로 토큰으로 시작하고 그 뒤에 추가 프롬프트 텍스트가 오는 경우 제출 시간 복구에 사용됩니다.

Args:
        text: 검사할 텍스트를 입력하세요.

Returns:
        유효한 선행 파일 경로 토큰이 없는 경우 `(resolved_path, token_end_index)` 또는 `None`의 튜플입니다.

    """
    if not text:
        return None

    start = len(text) - len(text.lstrip())
    payload = text[start:]
    token_end = _leading_token_end(payload)
    if token_end is None:
        return None

    token_text = payload[:token_end]
    path = parse_single_pasted_file_path(token_text)
    if path is None:
        spaced = _extract_unquoted_leading_path_with_spaces(payload)
        if spaced is None:
            return None
        spaced_path, spaced_end = spaced
        return spaced_path, start + spaced_end

    return path, start + token_end


def normalize_pasted_path(text: str) -> Path | None:
    """단일 파일 시스템 경로를 나타낼 수 있는 붙여넣은 텍스트를 정규화합니다.

    지원:

    - 인용 및 쉘 이스케이프 처리된 단일 경로 - `file://` URL - Windows 드라이브 문자 및 UNC 경로

Args:
        text: 원시 붙여넣은 텍스트 페이로드입니다.

Returns:
        페이로드가 단일 경로 토큰인 경우 `Path`, 그렇지 않은 경우 `None`을 구문 분석했습니다.

    """
    payload = text.strip()
    if not payload:
        return None

    unquoted = (
        payload.removeprefix('"').removesuffix('"')
        if payload.startswith('"') and payload.endswith('"')
        else payload
    )
    unquoted = (
        unquoted.removeprefix("'").removesuffix("'")
        if unquoted.startswith("'") and unquoted.endswith("'")
        else unquoted
    )

    if unquoted.startswith("file://"):
        return _token_to_path(unquoted)

    windows_path = _normalize_windows_pasted_path(unquoted)
    if windows_path is not None:
        return windows_path

    posix_path = _normalize_posix_pasted_path(unquoted)
    if posix_path is not None:
        return posix_path

    parts = _split_paste_line(payload)
    if len(parts) != 1:
        return None
    token = parts[0]
    path = _token_to_path(token)
    if path is None:
        return None
    windows_token_path = _normalize_windows_pasted_path(str(path))
    if windows_token_path is not None:
        return windows_token_path
    return path


def _split_paste_line(line: str) -> list[str]:
    """붙여넣은 단일 줄을 경로형 토큰으로 분할합니다.

Args:
        line: 붙여넣기 페이로드의 한 줄입니다.

Returns:
        구문 분석된 셸형 토큰 또는 구문 분석 실패 시 빈 목록입니다.

    """
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        # Unbalanced quotes or other tokenization errors: treat as plain text.
        return []


def _token_to_path(token: str) -> Path | None:
    """붙여넣은 토큰을 경로 후보로 변환합니다.

Args:
        token: 붙여넣기 페이로드의 단일 셸 분할 토큰입니다.

Returns:
        구문 분석된 경로 후보 또는 토큰 구문 분석이 실패할 경우 `None`입니다.

    """
    value = token.strip()
    if not value:
        return None

    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
        if not value:
            return None

    if value.startswith("file://"):
        parsed = urlparse(value)
        path_text = unquote(parsed.path or "")
        if parsed.netloc and parsed.netloc != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        if (
            path_text.startswith("/")
            and len(path_text) > 2  # noqa: PLR2004  # '/C:' minimum for Windows file URI
            and path_text[2] == ":"
            and path_text[1].isalpha()
        ):
            # `file:///C:/...` on Windows includes an extra leading slash.
            path_text = path_text[1:]
        if not path_text:
            return None
        return Path(path_text)

    return Path(value)


def _leading_token_end(text: str) -> int | None:
    """첫 번째 쉘형 토큰의 끝 인덱스를 반환합니다.

Args:
        text: 토큰으로 시작하는 텍스트를 입력하세요.

Returns:
        종료 인덱스(독점) 또는 토큰 구문 분석이 실패할 경우 `None`입니다.

    """
    if not text:
        return None

    if text[0] in {'"', "'"}:
        quote = text[0]
        escaped = False
        for index in range(1, len(text)):
            char = text[index]
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                return index + 1
            escaped = False
        return None

    escaped = False
    for index, char in enumerate(text):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char.isspace() and not escaped:
            return index
        escaped = False
    return len(text)


def _extract_unquoted_leading_path_with_spaces(text: str) -> tuple[Path, int] | None:
    """공백이 포함될 수 있는 따옴표가 없는 선행 경로를 추출합니다.

    이 폴백은 슬래시 명령 충돌이 `/`로 시작하는 입력에만 해당되기 때문에 의도적으로 POSIX 지향(`/` 및 `~/`)입니다.

Args:
        text: 잠재적인 경로로 시작하는 텍스트를 입력하세요.

Returns:
        일치하는 선행 경로 접두사가 기존 파일로 확인되지 않는 경우 `(resolved_path, token_end_index)` 또는 `None`의
        튜플입니다.

    """
    if not text or ("\n" in text or "\r" in text):
        return None
    if not text.startswith(("/", "~/")):
        return None
    if " " not in text and "\u00a0" not in text and "\u202f" not in text:
        return None

    boundaries = [index for index, char in enumerate(text) if char.isspace()]
    boundaries.append(len(text))
    for end in reversed(boundaries):
        candidate = text[:end].rstrip()
        if not candidate:
            continue
        path = parse_single_pasted_file_path(candidate)
        if path is not None:
            return path, len(candidate)
    return None


def _normalize_windows_pasted_path(text: str) -> Path | None:
    """따옴표가 없는 Windows 드라이브/UNC 경로 입력에 대해서는 `Path`을 반환합니다.

Args:
        text: 잠재적인 Windows 경로 입력.

Returns:
        `text`이 Windows 드라이브 문자 또는 UNC 스타일인 경우 `Path`을(를) 구문 분석하고, 그렇지 않으면 `None`을(를) 구문
        분석했습니다.

    """
    if _WINDOWS_DRIVE_PATH_PATTERN.match(text) or text.startswith("\\\\"):
        return Path(text)
    return None


def _normalize_posix_pasted_path(text: str) -> Path | None:
    """POSIX 절대/홈 경로 페이로드일 가능성이 있는 경우 `Path`을 반환합니다.

    일부 터미널에서는 공백이 포함된 삭제된 절대 경로를 인용/이스케이프 없이 원시 텍스트로 붙여넣습니다. 이 경우 전체 페이로드가 단일 경로가 되도록
    의도되었더라도 쉘 토큰화는 공백으로 분할됩니다.

Args:
        text: 잠재적인 POSIX 경로 입력.

Returns:
        `text`이 원시 POSIX 절대/홈 경로처럼 보이면 `Path`을 구문 분석하고, 그렇지 않으면 `None`을 구문 분석했습니다.

    """
    if "\n" in text or "\r" in text:
        return None
    if text.startswith("~/"):
        return Path(text)
    if text.startswith("/") and "/" in text[1:]:
        return Path(text)
    return None


def _resolve_existing_pasted_path(path: Path) -> Path | None:
    """붙여넣은 경로 후보를 기존 파일로 확인합니다.

    먼저 정확한 확인을 수행한 다음 유니코드 공간 허용 조회를 수행합니다.

Args:
        path: 구문 분석된 경로 후보.

Returns:
        기존 파일 경로를 확인했습니다. 그렇지 않으면 `None`입니다.

    """
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError) as e:
        logger.debug("Path resolution failed for %r: %s", path, e)
        return None
    if resolved.exists() and resolved.is_file():
        return resolved

    fuzzy = _resolve_with_unicode_space_variants(path)
    if fuzzy is None:
        return None
    try:
        resolved_fuzzy = fuzzy.resolve()
    except (OSError, RuntimeError) as e:
        logger.debug("Unicode-space resolution failed for %r: %s", fuzzy, e)
        return None
    if resolved_fuzzy.exists() and resolved_fuzzy.is_file():
        return resolved_fuzzy
    return None


def _normalize_unicode_spaces(text: str) -> str:
    """유니코드 유사 공간을 ASCII 공간으로 정규화합니다.

Args:
        text: 정규화할 텍스트입니다.

Returns:
        유니코드 공간 변형이 ASCII 공간으로 변환된 정규화된 텍스트입니다.

    """
    return text.translate(_UNICODE_SPACE_EQUIVALENTS)


def _resolve_with_unicode_space_variants(path: Path) -> Path | None:
    """파일 이름 세그먼트를 유니코드 공백 변형과 일치시켜 경로를 확인합니다.

Args:
        path: 공간 코드 포인트에 따라 디스크와 다를 수 있는 경로 후보입니다.

Returns:
        일치하는 파일 시스템 경로 또는 일치하는 변형이 없는 경우 `None`입니다.

    """
    expanded = path.expanduser()
    if expanded.is_absolute():
        current = Path(expanded.anchor)
        parts = expanded.parts[1:]
    else:
        current = Path.cwd()
        parts = expanded.parts

    for index, part in enumerate(parts):
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue

        if not current.exists() or not current.is_dir():
            return None
        if " " not in part and "\u00a0" not in part and "\u202f" not in part:
            return None

        normalized_part = _normalize_unicode_spaces(part)
        try:
            matches = [
                entry
                for entry in current.iterdir()
                if _normalize_unicode_spaces(entry.name) == normalized_part
            ]
        except OSError as e:
            logger.debug("Failed listing %s for Unicode-space lookup: %s", current, e)
            return None

        if not matches:
            return None

        is_last = index == len(parts) - 1
        if is_last:
            file_matches = [entry for entry in matches if entry.is_file()]
            if file_matches:
                matches = file_matches
        else:
            dir_matches = [entry for entry in matches if entry.is_dir()]
            if dir_matches:
                matches = dir_matches

        matches.sort(key=lambda entry: entry.name)
        current = matches[0]

    return current
