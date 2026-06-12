import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.config import TRAIN_DATA_PATH

documents = None
ids = None
metadatas = None
document_vectors = None
vectorizer = None


def load_incident_rows():
    df = pd.read_csv(TRAIN_DATA_PATH)

    documents = []
    ids = []
    metadatas = []

    for _, row in df.iterrows():
        doc = f"""
Incident:
{row["log"]}

Category:
{row["category"]}

Severity:
{row["severity"]}

Risk Level:
{row["risk_level"]}

Resolution:
Immediate Action: {row["immediate_action"]}
Short-Term Mitigation: {row["short_term_mitigation"]}
Long-Term Prevention: {row["long_term_prevention"]}
"""
        documents.append(doc)
        ids.append(str(row["id"]))
        metadatas.append(
            {
                "category": str(row["category"]),
                "risk_level": str(row["risk_level"]),
                "severity": str(row["severity"]),
            }
        )

    return documents, ids, metadatas


def initialize_memory():
    global documents
    global ids
    global metadatas
    global document_vectors
    global vectorizer

    if documents is None:
        documents, ids, metadatas = load_incident_rows()
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        document_vectors = vectorizer.fit_transform(documents)

    return len(documents)


def reset_memory():
    global documents
    documents = None
    return initialize_memory()


def search_incidents(query, n_results=3):
    results = search_incident_records(query, n_results=n_results)
    documents = results["documents"][0]
    return "\n\n--- Similar Incident ---\n\n".join(documents)


def search_incident_records(query, n_results=3):
    initialize_memory()
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, document_vectors)[0]
    ranked_indices = scores.argsort()[::-1][:n_results]

    return {
        "documents": [[documents[index] for index in ranked_indices]],
        "ids": [[ids[index] for index in ranked_indices]],
        "metadatas": [[metadatas[index] for index in ranked_indices]],
        "distances": [[float(1 - scores[index]) for index in ranked_indices]],
    }


initialize_memory()
