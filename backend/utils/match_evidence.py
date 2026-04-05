"""
Deterministic evidence checks after LLM matching.

Catches obvious matches (e.g. Python in CV) and role-implied QA/testing skills
when the model misses them.
"""

from __future__ import annotations

import re

# CV blob = full text + structured skills (skills often repeat key terms)
def _blob(cv_text: str | None, cv_skills: list[str]) -> str:
    parts = [cv_text or "", " ".join(cv_skills or [])]
    return "\n\n".join(parts)


def _qa_role_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"qa\b|quality assurance|sdet\b|test engineer|testing engineer|"
            r"software engineer in test|set engineer|qa engineer|test lead|test manager|"
            r"manual test|automated test|test automation|quality engineer|"
            r"software quality|verification engineer"
            r")\b",
            blob_lower,
        )
    )


def _ci_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"continuous integration|ci/cd|\bci\b|jenkins|gitlab ci|github actions|"
            r"circleci|travis|teamcity|bamboo|buildkite|azure devops pipelines"
            r")\b",
            blob_lower,
        )
        or "ci/cd" in blob_lower
    )


def _db_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"sql|mysql|postgres|postgresql|mongodb|redis|cassandra|oracle|"
            r"sqlite|dynamodb|elasticsearch|nosql|relational database|database\b"
            r")\b",
            blob_lower,
        )
    )


def _mapreduce_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"mapreduce|map-reduce|map reduce|hadoop|apache spark|\bspark\b|hive\b|pig\b|"
            r"big data|data pipeline|etl\b"
            r")\b",
            blob_lower,
        )
    )


def _storage_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"storage system|enterprise storage|san\b|nas\b|ssd\b|disk array|"
            r"block storage|file storage|volume manager|data store|storage area network"
            r")\b",
            blob_lower,
        )
        or ("storage" in blob_lower and re.search(r"\b(ssd|san|nas|disk|volume|lun)\b", blob_lower))
    )


def _distributed_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b("
            r"distributed system|microservices|kubernetes|\bk8s\b|kafka|rabbitmq|"
            r"multi-?node|cluster\b|consensus|raft\b|load balanc|scalability|"
            r"service mesh|grpc|rest api at scale"
            r")\b",
            blob_lower,
        )
    )


def _virt_esxi(blob_lower: str) -> bool:
    return bool(re.search(r"\b(vmware|esxi|vsphere|vcenter)\b", blob_lower))


def _virt_hyperv(blob_lower: str) -> bool:
    return bool(re.search(r"hyper-?v|microsoft hyper", blob_lower))


def _virt_kvm(blob_lower: str) -> bool:
    return bool(re.search(r"\bkvm\b|qemu|libvirt", blob_lower))


def _mentoring_signal(blob_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(mentor|mentoring|mentored|coach|coaching|coached|train(ed|ing)\s+(junior|team|engineers))\b",
            blob_lower,
        )
    )


def _python_signal(blob_lower: str) -> bool:
    return bool(re.search(r"\bpython\b", blob_lower))


def _go_signal(blob_lower: str) -> bool:
    if re.search(r"\bgolang\b", blob_lower):
        return True
    if re.search(r"\bgo\s+(programming|language|developer|lang)\b", blob_lower):
        return True
    if re.search(r"programming\s+in\s+go\b", blob_lower):
        return True
    # "Go" as a language token (narrow)
    if re.search(r"[\s,;/(]go[\s,;)/]", blob_lower):
        return True
    if re.search(r"\bgo\b", blob_lower) and re.search(
        r"\b(language|compiler|module|goroutine|golang)\b", blob_lower
    ):
        return True
    return False


# When QA/test role is established, these JD lines are treated as implied (no literal phrase required).
_QA_IMPLIED_NORMALIZED = frozenset(
    {
        "testing",
        "test planning",
        "test methodology",
        "test methodologies",
        "qa methodologies",
        "qa methodology",
        "test execution",
        "feature design",  # test/feature design in QA context
        "defect filing",
        "failure analysis",
        "root cause analysis",
    }
)


def _defect_signal(blob_lower: str) -> bool:
    return bool(re.search(r"\b(defect|jira|bugzilla|bug\s+track|triage|filed bugs)\b", blob_lower))


def _failure_rca_signal(blob_lower: str) -> bool:
    return bool(
        re.search(r"root\s+cause|rca\b|failure analysis|post-?mortem|postmortem", blob_lower)
    )


def skill_evidence(skill: str, blob: str, blob_lower: str) -> bool:
    """
    Return True if CV blob supports this JD skill by literal mention,
    alias, or (for QA-implied) role + related signals.
    """
    s = skill.strip()
    if not s:
        return False
    key = s.lower()
    # Normalize common punctuation in JD lists
    key_compact = re.sub(r"\s+", " ", key)

    # --- Literal phrase (multi-word JD items) ---
    if " " in key_compact or "-" in s:
        if key_compact in blob_lower:
            return True
        # "map-reduce frameworks" vs "map reduce"
        alts = [
            key_compact.replace("-", " "),
            key_compact.replace(" ", "-"),
        ]
        for a in alts:
            if a in blob_lower:
                return True

    qa = _qa_role_signal(blob_lower)

    # --- Languages (match JD labels like "Python", "programming in Python") ---
    if re.search(r"\bpython\b", key_compact):
        return _python_signal(blob_lower)
    if re.search(r"\bgolang\b", key_compact) or re.search(r"\bgo\b", key_compact):
        return _go_signal(blob_lower)

    # --- QA-implied (standard components of QA work) ---
    if qa and key_compact in _QA_IMPLIED_NORMALIZED:
        if key_compact == "defect filing":
            return _defect_signal(blob_lower) or qa
        if key_compact in ("failure analysis", "root cause analysis"):
            return _failure_rca_signal(blob_lower) or qa
        return True

    # --- Continuous integration ---
    if "continuous integration" in key or key == "ci/cd":
        return _ci_signal(blob_lower)

    # --- Mentoring ---
    if "mentor" in key:
        return _mentoring_signal(blob_lower)

    # --- Databases ---
    if "database" in key:
        return _db_signal(blob_lower)

    # --- MapReduce / big data ---
    if "map-reduce" in key or "mapreduce" in key.replace(" ", "") or "map reduce" in key:
        return _mapreduce_signal(blob_lower)

    # --- Storage / distributed ---
    if "storage" in key and "concept" in key:
        return _storage_signal(blob_lower)
    if "distributed" in key:
        return _distributed_signal(blob_lower)

    # --- Virtualization products (JD may say "ESXi", "knowledge of Hyper-V", etc.) ---
    kv = key_compact.replace(" ", "")
    if "esxi" in key or "vmware" in key or "vsphere" in key:
        return _virt_esxi(blob_lower)
    if re.search(r"hyper-?v", key_compact) or "hyperv" in kv:
        return _virt_hyperv(blob_lower)
    if "kvm" in key:
        return _virt_kvm(blob_lower)

    # --- Single-token fallback: word boundary ---
    if " " not in key_compact and len(key_compact) >= 2:
        # Avoid matching short tokens broadly
        if key_compact in ("go",):
            return _go_signal(blob_lower)
        return bool(re.search(r"\b" + re.escape(key_compact) + r"\b", blob_lower))

    return False


def augment_strengths_from_cv(
    jd_unique: list[str],
    strengths: list[str],
    cv_text: str | None,
    cv_skills: list[str],
) -> list[str]:
    """
    Add JD skills that have deterministic evidence in the CV but were missed by the LLM.
    Preserves jd_unique order.
    """
    blob = _blob(cv_text, cv_skills)
    blob_lower = blob.lower()
    have = set(strengths)
    for jd in jd_unique:
        if jd in have:
            continue
        if skill_evidence(jd, blob, blob_lower):
            have.add(jd)
    return [jd for jd in jd_unique if jd in have]
