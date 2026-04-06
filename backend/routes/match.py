"""Skill matching endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import MatchRequest, MatchResponse
from services.openai_service import match_skills_strict

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Match"])


@router.post("/match", response_model=MatchResponse)
async def match(body: MatchRequest) -> MatchResponse:
    try:
        jd = [s.strip() for s in body.jd_skills if s and str(s).strip()]
        cv = [s.strip() for s in body.cv_skills if s and str(s).strip()]
        return await match_skills_strict(
            jd, 
            cv, 
            cv_text=body.cv_text, 
            experience_summary=body.experience_summary, 
            jd_seniority=body.jd_seniority
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("match failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
