"""
vectorizer.py
-------------
Classic bag-of-words -> TF-IDF vectorization using scikit-learn.
No embeddings, no neural nets -- just term frequency-inverse document
frequency weighting over the preprocessed (lemmatized, stopword-free)
text.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing import preprocess


def build_tfidf_vectors(doc1: str, doc2: str, ngram_range=(1, 2)):
    """
    Preprocesses both documents, then fits a single TF-IDF vectorizer
    across both so the vector spaces are directly comparable, and
    returns (vector1, vector2, fitted_vectorizer).
    """
    processed_1 = preprocess(doc1)
    processed_2 = preprocess(doc2)

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
    tfidf_matrix = vectorizer.fit_transform([processed_1, processed_2])

    return tfidf_matrix[0:1], tfidf_matrix[1:2], vectorizer


def compute_cosine_similarity(doc1: str, doc2: str, ngram_range=(1, 2)) -> float:
    """
    Returns a similarity score in [0, 1] between two raw text documents,
    using TF-IDF vectors and cosine similarity between them.
    """
    vec1, vec2, _ = build_tfidf_vectors(doc1, doc2, ngram_range=ngram_range)
    score = cosine_similarity(vec1, vec2)[0][0]
    return float(score)


def top_tfidf_terms(text: str, top_n: int = 15):
    """
    Returns the top-N highest-weighted TF-IDF terms for a single document.
    Useful for showing 'what the model thinks this resume/JD is about'.
    """
    processed = preprocess(text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([processed])
    scores = matrix.toarray()[0]
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


if __name__ == "__main__":
    resume = "Skilled Python developer with experience in Django and machine learning."
    jd = "Looking for a Python developer familiar with Django and REST APIs."
    print("Cosine similarity:", compute_cosine_similarity(resume, jd))
    print("Top resume terms:", top_tfidf_terms(resume))
