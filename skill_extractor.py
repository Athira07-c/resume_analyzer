"""
skill_extractor.py
-------------------
Pure lexicon + n-gram matching (no ML) to pull out which known skills
appear in a piece of text. We generate 1-, 2-, and 3-word n-grams from
the *lemmatized* tokens and intersect them against the skills database,
so "problem solving" / "problem-solving" / "solved problems" all still
resolve correctly after lemmatization.
"""

from text_preprocessing import clean_text, tokenize, remove_stopwords, lemmatize
from skills_database import get_all_skills, get_skill_category


def _generate_ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_skills(text: str):
    """
    Returns a sorted list of (skill, category) tuples found in `text`.
    """
    all_skills = get_all_skills()

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens_no_stop = remove_stopwords(tokens)
    lemmas = lemmatize(tokens_no_stop)

    # Also lemmatize the *raw* tokens (pre-stopword-removal) for 2/3-grams
    # like "problem solving" where 'solving' matters but bigram order
    # depends on adjacency in the original sentence, not the stopword-
    # filtered stream. We build candidates from both to be safe.
    raw_lemmas = lemmatize(tokens)

    candidates = set(lemmas)
    for n in (2, 3):
        candidates.update(_generate_ngrams(lemmas, n))
        candidates.update(_generate_ngrams(raw_lemmas, n))

    found = set()
    for skill in all_skills:
        skill_lemma = " ".join(lemmatize(skill.split()))
        if skill in candidates or skill_lemma in candidates or skill in cleaned:
            found.add(skill)

    return sorted((skill, get_skill_category(skill)) for skill in found)


def extract_skill_names(text: str):
    return sorted(skill for skill, _ in extract_skills(text))


if __name__ == "__main__":
    sample = """
    Experienced software engineer skilled in Python, Django, and REST APIs.
    Strong background in machine learning, TensorFlow, and data visualization
    using Tableau. Excellent communication and problem-solving abilities,
    with hands-on AWS and Docker experience.
    """
    for skill, category in extract_skills(sample):
        print(f"{category:20s} -> {skill}")
