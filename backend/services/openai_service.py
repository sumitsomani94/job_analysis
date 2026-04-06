"""OpenAI API calls — lazy client initialization."""

import json
import logging
import os
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from models.schemas import (
    AnalyzeCVResponse,
    AnalyzeJDResponse,
    MatchResponse,
    SyllabusItem,
)
from utils.helpers import normalize_ai_text_field
from utils.match_evidence import augment_strengths_from_cv

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_TOKENS = 4096

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def _openai_friendly_error(exc: APIStatusError) -> RuntimeError:
    code = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code")
    if exc.status_code == 429:
        if code == "insufficient_quota":
            return RuntimeError(
                "OpenAI quota exceeded. Add billing or credits at "
                "https://platform.openai.com/account/billing — then retry."
            )
        return RuntimeError(
            "OpenAI rate limit reached. Wait a minute and try again, or check usage limits."
        )
    msg = getattr(exc, "message", None) or str(exc)
    return RuntimeError(f"OpenAI API error ({exc.status_code}): {msg}")


async def _chat_json(system: str, user: str) -> dict[str, Any]:
    client = _get_client()
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except APIStatusError as e:
        raise _openai_friendly_error(e) from e
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty response from OpenAI")
    return json.loads(content)


def _as_str_list(v: Any) -> list[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def _align_strengths_to_jd(jd_unique: list[str], claimed: list[str]) -> list[str]:
    """Map model output to canonical jd_skill strings (case-insensitive)."""
    norm_to_canon = {s.strip().lower(): s for s in jd_unique}
    out: list[str] = []
    seen: set[str] = set()
    for c in claimed:
        key = str(c).strip().lower()
        if key in norm_to_canon:
            canon = norm_to_canon[key]
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out


async def analyze_job_description(job_description: str) -> AnalyzeJDResponse:
    system = (
        "You extract structured data from job descriptions. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    user = f"""Analyze this job description and return JSON with this exact shape:
{{
  "skills": ["list of all distinct technical and professional skills mentioned"],
  "responsibilities": "string summarizing main responsibilities",
  "seniority_level": "one of: Junior, Mid, Senior, Lead, Principal, Executive, or Unknown",
  "categorized_skills": {{
    "must_have": ["skills explicitly required or must-have"],
    "good_to_have": ["nice-to-have or preferred skills"]
  }}
}}

Job description:
{job_description}
"""
    data = await _chat_json(system, user)
    responsibilities = normalize_ai_text_field(data.get("responsibilities"))
    must_have = _as_str_list(data.get("categorized_skills", {}).get("must_have"))
    good_to_have = _as_str_list(data.get("categorized_skills", {}).get("good_to_have"))
    
    # Merge all extracted skills ensuring nothing is dropped
    base_skills = _as_str_list(data.get("skills"))
    skills = list(dict.fromkeys(base_skills + must_have + good_to_have))
    
    seniority = str(data.get("seniority_level") or "Unknown").strip()
    return AnalyzeJDResponse(
        skills=skills,
        responsibilities=responsibilities,
        seniority_level=seniority,
        categorized_skills={"must_have": must_have, "good_to_have": good_to_have},
    )


async def analyze_cv_text(cv_text: str) -> AnalyzeCVResponse:
    system = (
        "You extract structured data from CV/resume text. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    user = f"""From this CV text, return JSON with this exact shape:
{{
  "skills": ["exhaustive list of skills evidenced ANYWHERE in the CV: summary, bullets, experience, projects"],
  "experience_summary": "brief string summary of professional experience",
  "domains": ["industry or domain areas mentioned"]
}}

Rules for skills:
- Include tools, languages, OS, virtualization (VMware/ESXi/Hyper-V/KVM if named), CI/CD, test types (manual, automated, API, performance), frameworks, databases, cloud, methodologies (Agile, QA practices), and domains (storage, distributed systems) when the CV supports them.
- If the CV describes QA/test/quality engineering roles with hands-on testing, also add standard competencies that such roles normally include: e.g. test planning, manual testing — when the narrative supports them (not if the role is purely non-testing).
- Use short job-style phrases where natural (e.g. "manual testing", "test automation", "Python", "Linux", "storage", "SSD" when relevant to work).
- Do not invent technologies never mentioned; paraphrases in the CV count (e.g. "automated UI tests" implies automated testing).

CV text:
{cv_text[:120000]}
"""
    data = await _chat_json(system, user)
    experience_summary = normalize_ai_text_field(data.get("experience_summary"))
    return AnalyzeCVResponse(
        skills=_as_str_list(data.get("skills")),
        experience_summary=experience_summary,
        domains=_as_str_list(data.get("domains")),
    )


async def match_skills_strict(
    jd_skills: list[str],
    cv_skills: list[str],
    cv_text: str | None = None,
) -> MatchResponse:
    system = (
        "You are an expert technical Engineering Manager and Senior Recruiter evaluating a candidate's CV against a Job Description. "
        "Your CORE directive is to use deep semantic inference. Candidates do not always use the exact keywords from the JD. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    jd_unique = list(dict.fromkeys(s.strip() for s in jd_skills if s and str(s).strip()))
    jd_json = json.dumps(jd_unique)
    cv_json = json.dumps(cv_skills)
    text_block = ""
    if cv_text and cv_text.strip():
        excerpt = cv_text.strip()[:80000]
        text_block = f"\n\nFULL CV TEXT (use for evidence; excerpt may be truncated):\n{excerpt}\n"

    user = f"""You must accurately match jd_skills to the candidate's CV for ANY industry (Tech, Finance, Healthcare, etc.).

CORE PRINCIPLE: INTELLIGENT INFERENCE & DEDUCTION
DO NOT just do dumb keyword matching. You must read the CV holistically to understand the candidate's actual daily work and experience.

1. INFERRED FOUNDATIONAL SKILLS (Crucial):
If a candidate has years of experience in a specialized role, you MUST infer the foundational competencies required to do that job.
- E.g. (Tech) If they build "automation frameworks" for "NVMe SSDs" using "Pytest", they ABSOLUTELY know "automated testing", "troubleshooting", and "storage validation".
- E.g. (Finance) If they are an "Auditor" who handles "SEC filings", they ABSOLUTELY have "financial reporting", "compliance", and "analytical skills".
- E.g. (Marketing) If they run "Google Ads CPA campaigns", they ABSOLUTELY know "digital marketing" and "data analysis".
You MUST mark these implicit foundational skills as strengths when analyzing their profile!

2. ROLE HIERARCHY & DOMAIN MATCHING:
If the JD asks for general domains like "systems products" or "distributed systems" or "QA methodologies", look at what the candidate actually does (e.g. SSDs, NVMe, CXL, Jenkins, Regression flows). These prove the domain. Mark it as a match.

3. STRICTNESS ON EXACT TECHNOLOGIES (Hard Negatives):
- You CANNOT infer purely distinct programming languages (e.g., Python does not mean they know Go). If JD asks for Go, and CV only has Python, mark Go as missing.
- You CANNOT infer specific vendor products like 'ESXi', 'Hyper-V', or 'KVM' unless virtualization at that exact layer is implied/mentioned. Mark them as missing if there are no virtualization signals.

4. EXHAUSTIVE OUTPUT:
- EVERY single JD skill MUST be categorized as either a strength OR a missing_skill. Do not drop any.

Return JSON:
{{
  "strengths": [],
  "missing_skills": []
}}

jd_skills: {jd_json}
cv_skills: {cv_json}
{text_block}
"""
    data = await _chat_json(system, user)
    strengths = _align_strengths_to_jd(jd_unique, _as_str_list(data.get("strengths")))
    strengths = augment_strengths_from_cv(
        jd_unique, strengths, cv_text=cv_text, cv_skills=cv_skills
    )
    strength_set = set(strengths)
    missing_skills = [s for s in jd_unique if s not in strength_set]
    n = len(jd_unique)
    if n == 0:
        pct = 100.0
    else:
        pct = (len(strengths) / n) * 100.0
    pct = max(0.0, min(100.0, pct))
    return MatchResponse(
        match_percentage=round(pct, 2),
        missing_skills=missing_skills,
        strengths=strengths,
    )


async def generate_syllabus(missing_skills: list[str]) -> list[SyllabusItem]:
    if not missing_skills:
        return []
    system = (
        "You create brief learning syllabi. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    user = f"""For each skill in this list, return BRIEF entries (3-4 subtopics, 2 practice questions each).

Return JSON:
{{
  "items": [
    {{
      "topic": "skill name",
      "subtopics": ["...", "..."],
      "difficulty": "Beginner|Intermediate|Advanced",
      "practice_questions": ["...", "..."]
    }}
  ]
}}

Skills: {json.dumps(missing_skills)}
"""
    data = await _chat_json(system, user)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        return []
    out: list[SyllabusItem] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        out.append(
            SyllabusItem(
                topic=str(it.get("topic") or "").strip(),
                subtopics=_as_str_list(it.get("subtopics"))[:8],
                difficulty=str(it.get("difficulty") or "Intermediate").strip(),
                practice_questions=_as_str_list(it.get("practice_questions"))[:4],
            )
        )
    return out


async def generate_interview_questions(
    job_description: str, missing_skills: list[str]
) -> list[str]:
    system = (
        "You write concise interview questions. "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    user = f"""Return exactly 5 concise interview questions tailored to the job and the candidate weak areas.

Return JSON:
{{
  "questions": ["q1", "q2", "q3", "q4", "q5"]
}}

Job description:
{job_description[:8000]}

Weak areas / missing skills:
{json.dumps(missing_skills)}
"""
    data = await _chat_json(system, user)
    qs = _as_str_list(data.get("questions"))[:5]
    defaults = [
        "Walk me through a recent project that aligns with this role.",
        "What gaps do you see between your background and this job description?",
        "How would you prioritize conflicting deadlines in this context?",
        "Describe how you stay current with tools and practices in this field.",
        "What questions do you have about the team or the expectations for this role?",
    ]
    i = 0
    while len(qs) < 5:
        qs.append(defaults[i % len(defaults)])
        i += 1
    return qs[:5]
