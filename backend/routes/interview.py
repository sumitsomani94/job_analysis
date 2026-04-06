"""Interview questions endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import InterviewRequest, InterviewResponse
from services.openai_service import generate_interview_questions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Interview"])


@router.post("/interview", response_model=InterviewResponse)
async def interview(body: InterviewRequest) -> InterviewResponse:
    try:
        missing = [s.strip() for s in body.missing_skills if s and str(s).strip()]
        qs = await generate_interview_questions(
            body.job_description or "", 
            missing, 
            experience_summary=body.experience_summary
        )
        return InterviewResponse(questions=qs)
    except Exception as e:  # noqa: BLE001
        logger.exception("interview failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
