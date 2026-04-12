"""
Unified AI Service Routing Layer.
Dynamically switches between Gemini and OpenAI depending on environment variable LLM_PROVIDER.
Defaults to openai if LLM_PROVIDER is not set to 'gemini'.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Determine which provider to use. If not specified, default to gemini if the API key exists, else openai.
provider = os.getenv("LLM_PROVIDER", "").lower().strip()

if not provider:
    if os.getenv("GEMINI_API_KEY"):
        provider = "gemini"
    else:
        provider = "openai"

logger.info(f"Using LLM Provider: {provider.upper()}")

if provider == "gemini":
    from services.gemini_service import (
        analyze_cv_text,
        analyze_job_description,
        generate_interview_questions,
        generate_syllabus,
        match_skills_strict,
    )
else:
    from services.openai_service import (
        analyze_cv_text,
        analyze_job_description,
        generate_interview_questions,
        generate_syllabus,
        match_skills_strict,
    )

__all__ = [
    "analyze_cv_text",
    "analyze_job_description",
    "generate_interview_questions",
    "generate_syllabus",
    "match_skills_strict",
]
