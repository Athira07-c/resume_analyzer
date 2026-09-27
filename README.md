# Resume Analyzer & Job Matcher (Pure NLP, No LLMs)

A classic, explainable NLP pipeline that scores how well a resume matches
a job description — no transformers, no embeddings from generative
models, no API calls. Everything is deterministic, inspectable, and
runs offline once set up.

## Pipeline

```
Raw text (resume / JD)
        │
        ▼
  clean_text()          lowercase, strip punctuation, normalize
        │                bullets, protect tokens like "c++"/"node.js"
        ▼
  word_tokenize()        NLTK tokenizer
        │
        ▼
  remove_stopwords()     drop "the", "and", "is"... (NLTK stopword list)
        │
        ▼
  pos_tag() + lemmatize()  POS-aware WordNet lemmatization
        │                  ("running"->"run", "developed"->"develop")
        ▼
  TF-IDF vectorization    scikit-learn TfidfVectorizer (uni+bi-grams)
        │
        ▼
  cosine_similarity()     similarity between resume & JD vectors
        │
        ▼
  skill_extractor.py      lexicon + n-gram matching against a curated
        │                 skills database (100+ skills, 7 categories)
        ▼
  matcher.py               combines TF-IDF similarity + skill coverage
                            into a final weighted score + gap report
```

## Project Structure

```
resume_analyzer/
├── text_preprocessing.py   # tokenize, remove stopwords, lemmatize
├── skills_database.py      # curated lexicon of skills by category
├── skill_extractor.py      # n-gram lexicon matching for skill extraction
├── vectorizer.py           # TF-IDF + cosine similarity
├── resume_parser.py        # extract text from PDF/DOCX/TXT
├── matcher.py              # combines everything into a final report
├── main.py                 # CLI entry point
├── app.py                  # Streamlit web UI
├── setup_nltk.py           # one-time NLTK corpus downloader
├── requirements.txt
└── sample_data/
    ├── sample_resume.txt
    └── sample_job_description.txt
```

## Setup

```bash
pip install -r requirements.txt
python setup_nltk.py   # downloads punkt, stopwords, wordnet, POS tagger
```

## Usage

### CLI

```bash
python main.py --resume sample_data/sample_resume.txt --jd sample_data/sample_job_description.txt
```

Add `--json` for machine-readable output:

```bash
python main.py --resume resume.pdf --jd jd.docx --json
```

### Web UI

```bash
streamlit run app.py
```

Upload or paste a resume and a job description, click **Analyze Match**,
and get:
- Overall match score (0-100%)
- TF-IDF text similarity
- Skill coverage %
- Matched / missing / extra skills
- A simple "years of experience" sanity check
- Top TF-IDF keywords for each document

### As a library

```python
from matcher import match_resume_to_job, format_report

result = match_resume_to_job(resume_text, jd_text)
print(format_report(result))
```

`result` is a plain dict:

```python
{
  "overall_match_score": 78.4,
  "tfidf_similarity": 62.1,
  "skill_match_percent": 94.7,
  "matched_skills": [...],
  "missing_skills": [...],
  "extra_skills_not_in_jd": [...],
  "experience_note": "Meets experience requirement (4 >= 3 years).",
  "top_resume_keywords": [...],
  "top_jd_keywords": [...],
}
```

## How the scoring works

`overall_match_score = 0.5 * TF-IDF_cosine_similarity + 0.5 * skill_coverage_ratio`

You can tune `WEIGHT_TFIDF_SIMILARITY` and `WEIGHT_SKILL_MATCH` in
`matcher.py` to favor one signal over the other.

- **TF-IDF similarity** captures overall topical/textual overlap
  (phrasing, domain vocabulary) between the two documents.
- **Skill coverage** is a more literal, explainable signal: what
  fraction of the skills explicitly required in the JD are found on
  the resume (using lemmatized n-gram matching, so "Problem-Solving"
  matches "solved problems", etc.).

## Extending it

- **More skills**: add phrases to `skills_database.py` — no retraining
  needed, it's just a lookup table.
- **Different weighting**: tweak the two `WEIGHT_*` constants in
  `matcher.py`.
- **Named entity extraction** (companies, degrees, dates): you could
  add `nltk.ne_chunk()` or regex rules in a new `entity_extractor.py`
  following the same pattern as `skill_extractor.py`.
- **Section-aware parsing**: split the resume into Education/Experience/
  Skills sections with regex/heading detection before running the
  pipeline per-section for more precise scoring.

## Why no LLMs/embeddings?

This project deliberately sticks to classic, interpretable NLP:
- Every step (tokens, lemmas, TF-IDF weights, matched skills) is
  inspectable — you can print intermediate output at any stage.
- No API costs, no external calls, fully offline after setup.
- Great for learning how NLP worked before transformer embeddings,
  and it's still a perfectly reasonable production approach for
  keyword/skill-based matching at scale.
