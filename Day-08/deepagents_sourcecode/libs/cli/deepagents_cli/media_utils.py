"""파일 및 클립보드의 이미지 및 비디오 입력을 표준화합니다.

이러한 도우미는 지원되는 미디어 소스를 감지하고 바이트를 안전하게 로드한 후 이를 에이전트 런타임에서 예상하는 메시지 콘텐츠 블록으로 변환합니다.
"""

import base64
import io
import logging
import os
import pathlib
import shutil

# S404: subprocess needed for clipboard access via pngpaste/osascript
import subprocess  # noqa: S404
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages.content import VideoContentBlock

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".ico",
    }
)
"""PIL이 지원하는 일반적인 이미지 파일 확장자."""

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".wmv",
    }
)
"""검증된 매직바이트 지원이 포함된 비디오 파일 확장자."""

MAX_MEDIA_BYTES: int = 20 * 1024 * 1024
"""최대 미디어 파일 크기(20MB). base64 페이로드를 ~27MB 미만으로 유지합니다."""


def _get_executable(name: str) -> str | None:
    """shutdown.which()를 사용하여 실행 파일의 전체 경로를 가져옵니다.

Args:
        name: 찾을 실행 파일의 이름

Returns:
        실행 파일의 전체 경로 또는 찾을 수 없는 경우 없음입니다.

    """
    return shutil.which(name)


@dataclass
class ImageData:
    """base64 인코딩으로 붙여넣은 이미지를 나타냅니다."""

    base64_data: str
    format: str  # "png", "jpeg", etc.
    placeholder: str  # Display text like "[image 1]"

    def to_message_content(self) -> dict:
        """LangChain 메시지 콘텐츠 형식으로 변환합니다.

Returns:
            다중 모드 메시지에 대한 유형 및 image_url이 포함된 사전입니다.

        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{self.format};base64,{self.base64_data}"},
        }


@dataclass
class VideoData:
    """base64 인코딩으로 붙여넣은 비디오를 나타냅니다."""

    base64_data: str
    format: str  # "mp4", "quicktime", etc.
    placeholder: str  # Display text like "[video 1]"

    def to_message_content(self) -> "VideoContentBlock":
        """LangChain `VideoContentBlock` 형식으로 변환합니다.

Returns:
            `VideoContentBlock`(base64 데이터 및 mime_type 포함)

        """
        from langchain_core.messages.content import create_video_block

        return create_video_block(
            base64=self.base64_data,
            mime_type=f"video/{self.format}",
        )


def get_clipboard_image() -> ImageData | None:
    """시스템 클립보드에서 이미지 읽기를 시도합니다.

    `pngpaste` 또는 `osascript`을 통해 macOS를 지원합니다.

Returns:
        이미지가 발견되면 ImageData이고, 그렇지 않으면 None입니다.

    """
    if sys.platform == "darwin":
        return _get_macos_clipboard_image()
    logger.warning(
        "Clipboard image paste is not supported on %s. "
        "Only macOS is currently supported. "
        "You can still attach images by dragging and dropping file paths.",
        sys.platform,
    )
    return None


def get_image_from_path(path: pathlib.Path) -> ImageData | None:
    """디스크에서 이미지 파일을 읽고 인코딩합니다.

Args:
        path: 이미지 파일의 경로입니다.

Returns:
        파일이 유효한 이미지인 경우 `ImageData`, 그렇지 않은 경우 `None`.

    """
    from PIL import Image, UnidentifiedImageError

    try:
        file_size = path.stat().st_size
        if file_size == 0:
            logger.debug("Image file is empty: %s", path)
            return None
        if file_size > MAX_MEDIA_BYTES:
            logger.warning(
                "Image file %s is too large (%d MB, max %d MB)",
                path,
                file_size // (1024 * 1024),
                MAX_MEDIA_BYTES // (1024 * 1024),
            )
            return None

        image_bytes = path.read_bytes()
        if not image_bytes:
            return None

        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()

        if image_format == "jpg":
            image_format = "jpeg"
        if not image_format:
            suffix = path.suffix.lower().removeprefix(".")
            image_format = "jpeg" if suffix == "jpg" else suffix
        if not image_format:
            image_format = "png"

        return ImageData(
            base64_data=encode_to_base64(image_bytes),
            format=image_format,
            placeholder="[image]",
        )
    except (UnidentifiedImageError, OSError) as e:
        logger.debug("Failed to load image from %s: %s", path, e, exc_info=True)
        return None


def _detect_video_format(data: bytes) -> str | None:
    """매직 바이트에서 비디오 MIME 하위 유형을 감지합니다.

Args:
        data: 원시 파일 바이트(신뢰할 수 있는 감지를 위해 최소 12바이트).

Returns:
        MIME 하위 유형(예: "mp4", "webm") 또는 인식할 수 없는 경우 `None`.

    """
    min_avi_len = 12
    if data[4:8] == b"ftyp":
        # ftyp box: major brand at bytes 8-12 distinguishes MOV vs MP4
        brand = data[8:12]
        if brand == b"qt  ":
            return "quicktime"
        return "mp4"
    if data[:4] == b"RIFF" and len(data) >= min_avi_len and data[8:12] == b"AVI ":
        return "avi"
    if data[:4] == b"\x30\x26\xb2\x75":  # ASF/WMV
        return "x-ms-wmv"
    if data[:4] == b"\x1a\x45\xdf\xa3":  # WebM/Matroska (EBML header)
        return "webm"
    return None


def get_video_from_path(path: pathlib.Path) -> VideoData | None:
    """디스크에서 비디오 파일을 읽고 인코딩합니다.

Args:
        path: 비디오 파일의 경로입니다.

Returns:
        파일이 유효한 비디오인 경우 `VideoData`, 그렇지 않은 경우 `None`.

    """
    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return None

    try:
        file_size = path.stat().st_size
        if file_size == 0:
            logger.debug("Video file is empty: %s", path)
            return None
        if file_size > MAX_MEDIA_BYTES:
            logger.warning(
                "Video file %s is too large (%d MB, max %d MB)",
                path,
                file_size // (1024 * 1024),
                MAX_MEDIA_BYTES // (1024 * 1024),
            )
            return None

        video_bytes = path.read_bytes()

        # Validate it's a real video file by checking magic bytes
        # MP4 starts with ftyp, MOV also uses ftyp, AVI starts with RIFF
        min_video_len = 8
        if len(video_bytes) < min_video_len:
            logger.debug("Video file too small (%d bytes): %s", len(video_bytes), path)
            return None

        # Detect format from magic bytes (not extension) so renamed files
        # get the correct MIME type.
        detected_format = _detect_video_format(video_bytes)
        if detected_format is None:
            logger.warning(
                "Video file %s has unrecognized signature for extension '%s'; "
                "skipping. If this is a valid video, the format may not be "
                "supported yet.",
                path,
                suffix,
            )
            return None

        return VideoData(
            base64_data=encode_to_base64(video_bytes),
            format=detected_format,
            placeholder="[video]",
        )
    except OSError as e:
        logger.warning("Failed to load video from %s: %s", path, e, exc_info=True)
        return None


def get_media_from_path(path: pathlib.Path) -> ImageData | VideoData | None:
    """먼저 파일을 이미지로 로드한 다음 비디오로 로드해 보십시오.

Args:
        path: 미디어 파일의 경로입니다.

Returns:
        파일이 유효한 미디어인 경우 `ImageData` 또는 `VideoData`, 그렇지 않은 경우 `None`.

    """
    result: ImageData | VideoData | None = get_image_from_path(path)
    if result is not None:
        return result
    return get_video_from_path(path)


def _get_macos_clipboard_image() -> ImageData | None:
    """pngpaste 또는 osascript를 사용하여 macOS에서 클립보드 이미지를 가져옵니다.

    먼저 pngpaste를 시도한 다음(설치하면 더 빠름) osascript로 돌아갑니다.

Returns:
        이미지가 발견되면 ImageData이고, 그렇지 않으면 None입니다.

    """
    from PIL import Image, UnidentifiedImageError

    # Try pngpaste first (fast if installed)
    pngpaste_path = _get_executable("pngpaste")
    if pngpaste_path:
        try:
            # S603: pngpaste_path is validated via shutil.which(), args are hardcoded
            result = subprocess.run(  # noqa: S603
                [pngpaste_path, "-"],
                capture_output=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout:
                # Successfully got PNG data - validate it's a real image
                try:
                    Image.open(io.BytesIO(result.stdout))
                    base64_data = base64.b64encode(result.stdout).decode("utf-8")
                    return ImageData(
                        base64_data=base64_data,
                        format="png",  # 'pngpaste -' always outputs PNG
                        placeholder="[image]",
                    )
                except (
                    # UnidentifiedImageError: corrupted or non-image data
                    UnidentifiedImageError,
                    OSError,  # OSError: I/O errors during image processing
                ) as e:
                    logger.debug(
                        "Invalid image data from pngpaste: %s", e, exc_info=True
                    )
        except FileNotFoundError:
            # pngpaste not installed - expected on systems without it
            logger.debug("pngpaste not found, falling back to osascript")
        except subprocess.TimeoutExpired:
            logger.debug("pngpaste timed out after 2 seconds")

    # Fallback to osascript with temp file (built-in but slower)
    return _get_clipboard_via_osascript()


def _get_clipboard_via_osascript() -> ImageData | None:
    """임시 파일을 사용하여 osascript를 통해 클립보드 이미지를 가져옵니다.

    osascript는 원시 바이너리로 캡처할 수 없는 특수 형식으로 데이터를 출력하므로 대신 임시 파일에 씁니다.

Returns:
        이미지가 발견되면 ImageData이고, 그렇지 않으면 None입니다.

    """
    from PIL import Image, UnidentifiedImageError

    # Get osascript path - it's a macOS builtin so should always exist
    osascript_path = _get_executable("osascript")
    if not osascript_path:
        return None

    # Create a temp file for the image
    fd, temp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    try:
        # First check if clipboard has PNG data
        # S603: osascript_path is validated via shutil.which(), args are hardcoded
        check_result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", "clipboard info"],
            capture_output=True,
            check=False,
            timeout=2,
            text=True,
        )

        if check_result.returncode != 0:
            return None

        # Check for PNG or TIFF in clipboard info
        clipboard_info = check_result.stdout.lower()
        if "pngf" not in clipboard_info and "tiff" not in clipboard_info:
            return None

        # Try to get PNG first, fall back to TIFF
        if "pngf" in clipboard_info:
            get_script = f"""
            set pngData to the clipboard as «class PNGf»
            set theFile to open for access POSIX file "{temp_path}" with write permission
            write pngData to theFile
            close access theFile
            return "success"
            """  # noqa: E501
        else:
            get_script = f"""
            set tiffData to the clipboard as TIFF picture
            set theFile to open for access POSIX file "{temp_path}" with write permission
            write tiffData to theFile
            close access theFile
            return "success"
            """  # noqa: E501

        # S603: osascript_path validated via shutil.which(), script is internal
        result = subprocess.run(  # noqa: S603
            [osascript_path, "-e", get_script],
            capture_output=True,
            check=False,
            timeout=3,
            text=True,
        )

        if result.returncode != 0 or "success" not in result.stdout:
            return None

        # Check if file was created and has content
        if (
            not pathlib.Path(temp_path).exists()
            or pathlib.Path(temp_path).stat().st_size == 0
        ):
            return None

        # Read and validate the image
        image_data = pathlib.Path(temp_path).read_bytes()

        try:
            image = Image.open(io.BytesIO(image_data))
            # Convert to PNG if it's not already (e.g., if we got TIFF)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return ImageData(
                base64_data=base64_data,
                format="png",
                placeholder="[image]",
            )
        except (
            # UnidentifiedImageError: corrupted or non-image data
            UnidentifiedImageError,
            OSError,  # OSError: I/O errors during image processing
        ) as e:
            logger.debug(
                "Failed to process clipboard image via osascript: %s", e, exc_info=True
            )
            return None

    except subprocess.TimeoutExpired:
        logger.debug("osascript timed out while accessing clipboard")
        return None
    except OSError as e:
        logger.debug("OSError accessing clipboard via osascript: %s", e)
        return None
    finally:
        # Clean up temp file
        try:
            pathlib.Path(temp_path).unlink()
        except OSError as e:
            logger.debug("Failed to clean up temp file %s: %s", temp_path, e)


def encode_to_base64(data: bytes) -> str:
    """원시 바이트를 base64 문자열로 인코딩합니다.

Args:
        data: 인코딩할 원시 바이트입니다.

Returns:
        Base64로 인코딩된 문자열.

    """
    return base64.b64encode(data).decode("utf-8")


def create_multimodal_content(
    text: str, images: list[ImageData], videos: list[VideoData] | None = None
) -> list[dict]:
    """텍스트, 이미지, 비디오로 다양한 메시지 콘텐츠를 만드세요.

Args:
        text: 메시지의 텍스트 내용
        images: ImageData 객체 목록
        videos: VideoData 개체의 선택적 목록

Returns:
        LangChain 메시지 형식의 콘텐츠 블록 목록입니다.

    """
    content_blocks = []

    # Add text block
    if text.strip():
        content_blocks.append({"type": "text", "text": text})

    # Add image blocks
    content_blocks.extend(image.to_message_content() for image in images)

    # Add video blocks
    if videos:
        content_blocks.extend(video.to_message_content() for video in videos)

    return content_blocks
