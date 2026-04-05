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
    skills = _as_str_list(data.get("skills"))
    if not skills:
        skills = list(dict.fromkeys(must_have + good_to_have))
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
        "You compare JD skills to a candidate's CV for ANY industry. "
        "When a CV establishes a parent role or domain, treat standard sub-skills as satisfied WITHOUT "
        "requiring those exact phrases in the CV (role hierarchy / subset reasoning). "
        "Still require real evidence for the parent; never equate unrelated technologies (e.g. Python is not Go). "
        "Return ONLY valid JSON. No markdown, no explanation."
    )
    jd_unique = list(dict.fromkeys(s.strip() for s in jd_skills if s and str(s).strip()))
    jd_json = json.dumps(jd_unique)
    cv_json = json.dumps(cv_skills)
    text_block = ""
    if cv_text and cv_text.strip():
        excerpt = cv_text.strip()[:80000]
        text_block = f"\n\nFULL CV TEXT (use for evidence; excerpt may be truncated):\n{excerpt}\n"

    user = f"""For each jd_skill, decide if the candidate has enough evidence — including IMPLIED evidence via role subsets (below).

CORE PRINCIPLE — ROLE HIERARCHIES (any candidate, any resume):
- Do NOT require the JD phrase to appear verbatim in the CV when it is a **standard sub-skill of a parent role/domain** that the CV **clearly establishes** with real experience (titles, responsibilities, projects).
- First identify **what parent competency** the CV proves (e.g. QA engineer, SDET, backend developer, SRE, data engineer, Linux admin). Then, if a jd_skill is a **normal, widely-expected component** of that parent in industry practice, count it as a STRENGTH even if the CV never uses those exact words.
- Only use this when the CV shows **meaningful** work in that parent area — not a single keyword with no context.

EXAMPLES OF SUBSET REASONING (non-exhaustive; apply the same logic to other roles):
- **QA / quality / test engineering** (hands-on testing): Subsumes typical JD items such as **test planning**, **test execution**, **manual testing**, **test cases**, **defect/triage** work, and **troubleshooting** *in a testing/quality context* — without requiring each phrase in the CV.
- **Software engineering / development**: Often subsumes **debugging**, **troubleshooting** code/systems, **code review**-adjacent skills when dev work is described.
- **SRE / DevOps / platform**: Often subsumes **CI/CD**, **monitoring**, **incident response**, **troubleshooting** production — when that role is evidenced.
- **Backend / API development**: Often subsumes **API**-related JD items when backend work is clear.
- **Data engineering**: Often subsumes **ETL/pipeline**-style JD items when that role is evidenced.
- **Linux admin / Linux-heavy roles**: Subsumes **Linux**, often **shell scripting** when CLI/bash work is implied.
- **Domain-specific**: **Storage**, **distributed systems**, **virtualization** products: apply subset rules only when the CV shows work in that domain; see HARD NEGATIVES.

A) DIRECT / SYNONYM MATCHES (always apply):
- Explicit names, synonyms, paraphrases (e.g. manual QA ↔ manual testing; CI/CD ↔ continuous integration; RHEL/Ubuntu ↔ Linux; bash/sh ↔ shell scripting).

B) DOMAIN-SPECIFIC EVIDENCE (when subset alone is not enough):
- **Storage concepts**: Need professional signals (SSDs, SAN/NAS, storage systems, disk performance, etc.), not casual hardware use.
- **Distributed systems**: Need real signals (clusters, microservices, Kafka, multi-node, etc.).
- **Programming in Go / Golang**: Require Go/Golang (or unmistakable Go-only stack) — **Python ≠ Go**.
- **ESXi / Hyper-V / KVM / VMware**: Require virtualization product evidence — do not infer from Linux alone.

C) HARD NEGATIVES:
- Never map one programming language to another.
- Do not invent employers, dates, or tools not supported by the CV.
- Do not mark a niche product skill from a generic OS/cloud mention alone.

OUTPUT:
- strengths: each matched item MUST be an EXACT string copy from jd_skills (same spelling/casing as in jd_skills list).
- missing_skills: jd_skills items not matched (exact strings from jd_skills).
- strengths + missing_skills must partition jd_skills (each jd skill appears exactly once).

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
