"""Learning syllabus endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import SyllabusItem, SyllabusRequest
from services.openai_service import generate_syllabus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Syllabus"])


@router.post("/syllabus", response_model=list[SyllabusItem])
async def syllabus(body: SyllabusRequest) -> list[SyllabusItem]:
    try:
        skills = [s.strip() for s in body.missing_skills if s and str(s).strip()]
        return await generate_syllabus(skills)
    except Exception as e:  # noqa: BLE001
        logger.exception("syllabus failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
