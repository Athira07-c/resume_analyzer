"""
matcher.py
----------
Ties the pipeline together:

    resume text, job description text
        -> TF-IDF cosine similarity   (overall textual match)
        -> lexicon-based skill match  (explicit skill gap analysis)
        -> weighted final score + human-readable report

Everything here is deterministic, rule-based NLP -- no LLM calls.
"""

import re
from vectorizer import compute_cosine_similarity, top_tfidf_terms
from skill_extractor import extract_skills

# How much weight each signal gets in the final score.
WEIGHT_TFIDF_SIMILARITY = 0.5
WEIGHT_SKILL_MATCH = 0.5


def _extract_years_of_experience(text: str):
    """Very simple regex heuristic: looks for patterns like '5 years',
    '3+ years of experience', etc. Not a substitute for careful parsing,
    just a bonus signal."""
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text.lower())
    years = [int(m) for m in matches]
    return max(years) if years else None


def match_resume_to_job(resume_text: str, jd_text: str) -> dict:
    # 1. Overall semantic-ish similarity via TF-IDF + cosine similarity
    similarity_score = compute_cosine_similarity(resume_text, jd_text)

    # 2. Skill extraction from both documents
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    resume_skill_names = {s for s, _ in resume_skills}
    jd_skill_names = {s for s, _ in jd_skills}

    matched_skills = sorted(resume_skill_names & jd_skill_names)
    missing_skills = sorted(jd_skill_names - resume_skill_names)
    extra_skills = sorted(resume_skill_names - jd_skill_names)

    skill_match_ratio = (
        len(matched_skills) / len(jd_skill_names) if jd_skill_names else 0.0
    )

    # 3. Weighted combined score (0-100)
    final_score = round(
        (WEIGHT_TFIDF_SIMILARITY * similarity_score
         + WEIGHT_SKILL_MATCH * skill_match_ratio) * 100,
        2,
    )

    # 4. Bonus heuristics
    years_required = _extract_years_of_experience(jd_text)
    years_found = _extract_years_of_experience(resume_text)

    experience_flag = None
    if years_required is not None:
        if years_found is None:
            experience_flag = f"JD asks for {years_required}+ years; couldn't detect experience in resume."
        elif years_found >= years_required:
            experience_flag = f"Meets experience requirement ({years_found} >= {years_required} years)."
        else:
            experience_flag = f"Below stated requirement ({years_found} < {years_required} years)."

    return {
        "overall_match_score": final_score,
        "tfidf_similarity": round(similarity_score * 100, 2),
        "skill_match_percent": round(skill_match_ratio * 100, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills_not_in_jd": extra_skills,
        "experience_note": experience_flag,
        "top_resume_keywords": [t for t, _ in top_tfidf_terms(resume_text, 10)],
        "top_jd_keywords": [t for t, _ in top_tfidf_terms(jd_text, 10)],
    }


def format_report(result: dict) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append(" RESUME <-> JOB DESCRIPTION MATCH REPORT")
    lines.append("=" * 55)
    lines.append(f"Overall Match Score : {result['overall_match_score']}%")
    lines.append(f"  - TF-IDF text similarity : {result['tfidf_similarity']}%")
    lines.append(f"  - Skill coverage         : {result['skill_match_percent']}%")
    lines.append("")
    lines.append(f"Matched Skills ({len(result['matched_skills'])}):")
    lines.append("  " + (", ".join(result["matched_skills"]) or "none"))
    lines.append("")
    lines.append(f"Missing Skills ({len(result['missing_skills'])}) -- consider adding these:")
    lines.append("  " + (", ".join(result["missing_skills"]) or "none"))
    lines.append("")
    lines.append(f"Extra Skills on Resume (not asked for in JD):")
    lines.append("  " + (", ".join(result["extra_skills_not_in_jd"]) or "none"))
    lines.append("")
    if result["experience_note"]:
        lines.append(f"Experience: {result['experience_note']}")
        lines.append("")
    lines.append(f"Top Resume Keywords : {', '.join(result['top_resume_keywords'])}")
    lines.append(f"Top JD Keywords     : {', '.join(result['top_jd_keywords'])}")
    lines.append("=" * 55)
    return "\n".join(lines)


if __name__ == "__main__":
    resume = """
    Software engineer with 4 years of experience building REST APIs using
    Python, Django, and PostgreSQL. Familiar with Docker, Git, and AWS.
    Strong communication and teamwork skills.
    """
    jd = """
    We are looking for a backend engineer with 3+ years of experience in
    Python and Django, strong knowledge of REST APIs, PostgreSQL, and
    AWS. Experience with Kubernetes and leadership skills are a plus.
    """
    result = match_resume_to_job(resume, jd)
    print(format_report(result))
