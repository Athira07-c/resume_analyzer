"""
Run this ONCE after `pip install -r requirements.txt` to download the
NLTK corpora this project needs (tokenizer, stopwords, POS tagger,
WordNet for lemmatization).

    python setup_nltk.py
"""
import nltk

RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

if __name__ == "__main__":
    for resource in RESOURCES:
        try:
            nltk.download(resource)
        except Exception as e:
            print(f"Skipping {resource}: {e}")
    print("\nNLTK setup complete.")
