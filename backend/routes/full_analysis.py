"""Unified full analysis pipeline."""

import logging
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.schemas import (
    AnalyzeCVResponse,
    AnalyzeJDResponse,
    FullAnalysisResponse,
    InterviewResponse,
    MatchResponse,
)
from services.openai_service import (
    analyze_cv_text,
    analyze_job_description,
    generate_interview_questions,
    generate_syllabus,
    match_skills_strict,
)
from services.pdf_service import extract_pdf_text_from_bytes, validate_pdf_upload
from services.session_store import save_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Full Analysis"])


@router.post("/analyze/full", response_model=FullAnalysisResponse)
async def analyze_full(
    job_description: str = Form(...),
    file: UploadFile = File(...),
) -> FullAnalysisResponse:
    jd_text = (job_description or "").strip()
    if not jd_text:
        raise HTTPException(status_code=400, detail="job_description is required.")

    try:
        raw = await file.read()
        validate_pdf_upload(file.content_type, len(raw))
        pdf_text = extract_pdf_text_from_bytes(raw)
        if not pdf_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF. Try a text-based PDF.",
            )

        jd_analysis: AnalyzeJDResponse = await analyze_job_description(jd_text)
        cv_analysis: AnalyzeCVResponse = await analyze_cv_text(pdf_text)

        jd_skills = list(dict.fromkeys(s.strip() for s in jd_analysis.skills if s.strip()))
        cv_skills = list(dict.fromkeys(s.strip() for s in cv_analysis.skills if s.strip()))

        match_result: MatchResponse = await match_skills_strict(
            jd_skills, cv_skills, cv_text=pdf_text
        )

        syllabus_items = await generate_syllabus(match_result.missing_skills)

        questions = await generate_interview_questions(jd_text, match_result.missing_skills)
        interview = InterviewResponse(questions=questions)

        session_id = str(uuid4())
        full = FullAnalysisResponse(
            session_id=session_id,
            jd_analysis=jd_analysis,
            cv_analysis=cv_analysis,
            match=match_result,
            syllabus=syllabus_items,
            interview=interview,
        )
        save_session(full.model_dump(mode="json"), session_id=session_id)
        return full
    except HTTPException:
        raise
    except ValueError as e:
        if "too large" in str(e).lower():
            raise HTTPException(status_code=413, detail=str(e)) from e
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("analyze_full failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
