"""PDF text extraction with pdfplumber."""

import io
import logging
from typing import BinaryIO

import pdfplumber

logger = logging.getLogger(__name__)

MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def validate_pdf_upload(content_type: str | None, size: int) -> None:
    """Raise ValueError if upload is invalid."""
    if content_type != "application/pdf":
        raise ValueError("Only application/pdf is accepted.")
    if size > MAX_BYTES:
        raise ValueError(f"File too large. Maximum size is {MAX_BYTES // (1024 * 1024)} MB.")


def extract_pdf_text_from_bytes(raw: bytes) -> str:
    """Extract plain text from PDF bytes."""
    if len(raw) > MAX_BYTES:
        raise ValueError(f"File too large. Maximum size is {MAX_BYTES // (1024 * 1024)} MB.")

    buf = io.BytesIO(raw)
    parts: list[str] = []
    try:
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
                except Exception as e:  # noqa: BLE001
                    logger.warning("pdf page extract failed: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.exception("pdf open failed: %s", e)
        raise ValueError("Could not read PDF. The file may be corrupt.") from e

    return "\n".join(parts).strip()


def extract_pdf_text(file: BinaryIO) -> str:
    """Extract plain text from a file-like object (reads all bytes)."""
    raw = file.read()
    return extract_pdf_text_from_bytes(raw)
