"""Job description analysis endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from models.schemas import AnalyzeJDRequest, AnalyzeJDResponse
from services.ai_service import analyze_job_description

logger = logging.getLogger(__name__)

router = APIRouter(tags=["JD"])


@router.post("/analyze/jd", response_model=AnalyzeJDResponse)
async def analyze_jd(body: AnalyzeJDRequest) -> AnalyzeJDResponse:
    try:
        return await analyze_job_description(body.job_description.strip())
    except Exception as e:  # noqa: BLE001
        logger.exception("analyze_jd failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
