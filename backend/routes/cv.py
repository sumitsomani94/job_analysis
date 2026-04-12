"""CV PDF analysis endpoint."""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.schemas import AnalyzeCVResponse
from services.ai_service import analyze_cv_text
from services.pdf_service import extract_pdf_text_from_bytes, validate_pdf_upload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["CV"])


@router.post("/analyze/cv", response_model=AnalyzeCVResponse)
async def analyze_cv(file: UploadFile = File(...)) -> AnalyzeCVResponse:
    try:
        raw = await file.read()
        validate_pdf_upload(file.content_type, len(raw))
        text = extract_pdf_text_from_bytes(raw)
        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Try a text-based PDF.",
            )
        return await analyze_cv_text(text)
    except HTTPException:
        raise
    except ValueError as e:
        if "too large" in str(e).lower():
            raise HTTPException(status_code=413, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("analyze_cv failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
