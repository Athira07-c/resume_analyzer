"""
text_preprocessing.py
----------------------
The classic NLP pipeline, step by step:

    raw text -> clean text -> tokenize -> remove stopwords
             -> POS-tag -> lemmatize -> cleaned tokens

No embeddings, no transformers -- just rule-based / lexicon-based NLP.
"""

import os
import re
import shutil
import string
from pathlib import Path

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


_PROJECT_ROOT = Path(__file__).resolve().parent
_NLTK_DATA_DIR = _PROJECT_ROOT / "nltk_data"
if str(_NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(_NLTK_DATA_DIR))
os.environ.setdefault("NLTK_DATA", str(_NLTK_DATA_DIR))


def _remove_partial_nltk_resource(resource_name: str):
    """Clear a fragmented NLTK install so a clean re-download can succeed."""
    mapping = {
        "punkt": Path("tokenizers") / "punkt",
        "punkt_tab": Path("tokenizers") / "punkt_tab",
        "stopwords": Path("corpora") / "stopwords",
        "wordnet": Path("corpora") / "wordnet",
        "omw-1.4": Path("corpora") / "omw-1.4",
        "averaged_perceptron_tagger": Path("taggers") / "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": Path("taggers") / "averaged_perceptron_tagger_eng",
    }
    target = mapping.get(resource_name)
    if not target:
        return

    for base in map(Path, nltk.data.path):
        candidate = base / target
        if candidate.exists():
            shutil.rmtree(candidate, ignore_errors=True)


def _resource_exists(resource_name: str) -> bool:
    """Check whether an NLTK resource is usable without triggering the broken path exception."""
    mapping = {
        "punkt": Path("tokenizers") / "punkt",
        "punkt_tab": Path("tokenizers") / "punkt_tab",
        "stopwords": Path("corpora") / "stopwords",
        "wordnet": Path("corpora") / "wordnet",
        "omw-1.4": Path("corpora") / "omw-1.4",
        "averaged_perceptron_tagger": Path("taggers") / "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": Path("taggers") / "averaged_perceptron_tagger_eng",
    }

    target = mapping.get(resource_name)
    if not target:
        return False

    for base in map(Path, nltk.data.path):
        candidate = base / target
        if candidate.exists():
            if resource_name == "punkt" and not (candidate / "PY3").exists():
                return False
            if resource_name == "punkt_tab" and not (candidate / "PY3_tab").exists():
                return False
            return True

    return False


def _ensure_nltk_data():
    """Download and repair the corpora needed by this app into the project-local NLTK cache."""
    _NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        if not _resource_exists(resource):
            _remove_partial_nltk_resource(resource)
            try:
                nltk.download(resource, download_dir=str(_NLTK_DATA_DIR), quiet=True)
            except Exception:
                pass


_ensure_nltk_data()

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))

# Words that look like stopwords but matter for resumes ("c", "r", "go" etc.
# are skill names in disguise). We keep single-char tokens out separately,
# so no special casing needed here, but we DO keep some words NLTK marks
# as stopwords because they can be meaningful in a skills context.
_KEEP_EVEN_IF_STOPWORD = {"c"}  # 'C' the language would otherwise vanish


def clean_text(text: str) -> str:
    """
    Lowercase, expand a few common resume abbreviations, strip out
    punctuation/numbers noise while preserving tokens like 'c++', 'c#',
    'node.js' that matter for skill matching.
    """
    text = text.lower()

    # normalize common separators / bullets
    text = re.sub(r"[•●▪◦‣·]", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)

    # protect a few special tech tokens before stripping punctuation
    protected = {
        "c++": "cpluplus_token",
        "c#": "csharp_token",
        "node.js": "nodejs_token",
        ".net": "dotnet_token",
    }
    for real, placeholder in protected.items():
        text = text.replace(real, placeholder)

    # remove punctuation except hyphens inside words (e.g. "problem-solving")
    text = re.sub(r"[^\w\s\-]", " ", text)

    # restore protected tokens
    reverse = {v: k for k, v in protected.items()}
    for placeholder, real in reverse.items():
        text = text.replace(placeholder, real)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wordnet_pos(nltk_pos_tag: str):
    """Map NLTK's POS tag to a WordNet POS tag for accurate lemmatization."""
    if nltk_pos_tag.startswith("J"):
        return wordnet.ADJ
    if nltk_pos_tag.startswith("V"):
        return wordnet.VERB
    if nltk_pos_tag.startswith("N"):
        return wordnet.NOUN
    if nltk_pos_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def tokenize(text: str):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [
        t for t in tokens
        if (t not in _STOPWORDS or t in _KEEP_EVEN_IF_STOPWORD)
        and t not in string.punctuation
        and len(t) > 1
    ]


def lemmatize(tokens):
    """POS-aware lemmatization: 'running' -> 'run', 'developed' -> 'develop'."""
    tagged = pos_tag(tokens)
    return [_LEMMATIZER.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged]


def preprocess(text: str, return_string: bool = True):
    """
    Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize.
    Returns a space-joined string (ready for TF-IDF) by default, or the
    raw list of lemmatized tokens if return_string=False.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize(tokens)
    return " ".join(lemmas) if return_string else lemmas


if __name__ == "__main__":
    sample = "Developed scalable REST APIs using Node.js and led a team of 4 engineers."
    print("Original :", sample)
    print("Processed:", preprocess(sample))
