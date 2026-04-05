"""All Pydantic models for request/response bodies."""

from pydantic import BaseModel, Field


class AnalyzeJDRequest(BaseModel):
    job_description: str = Field(..., min_length=1)


class AnalyzeJDResponse(BaseModel):
    skills: list[str] = Field(default_factory=list)
    responsibilities: str = ""
    seniority_level: str = ""
    categorized_skills: dict[str, list[str]] = Field(
        default_factory=lambda: {"must_have": [], "good_to_have": []}
    )


class AnalyzeCVResponse(BaseModel):
    skills: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    domains: list[str] = Field(default_factory=list)


class MatchRequest(BaseModel):
    jd_skills: list[str] = Field(default_factory=list)
    cv_skills: list[str] = Field(default_factory=list)
    cv_text: str | None = Field(
        default=None,
        description="Optional full CV/resume text for evidence-based matching.",
    )


class MatchResponse(BaseModel):
    match_percentage: float = 0.0
    missing_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)


class SyllabusRequest(BaseModel):
    missing_skills: list[str] = Field(default_factory=list)


class SyllabusItem(BaseModel):
    topic: str = ""
    subtopics: list[str] = Field(default_factory=list)
    difficulty: str = ""
    practice_questions: list[str] = Field(default_factory=list)


class InterviewRequest(BaseModel):
    job_description: str = ""
    missing_skills: list[str] = Field(default_factory=list)


class InterviewResponse(BaseModel):
    questions: list[str] = Field(default_factory=list)


class FullAnalysisResponse(BaseModel):
    session_id: str
    jd_analysis: AnalyzeJDResponse
    cv_analysis: AnalyzeCVResponse
    match: MatchResponse
    syllabus: list[SyllabusItem]
    interview: InterviewResponse
