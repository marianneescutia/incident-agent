import os
import pandas as pd
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "sample_data", "incidents.csv")

client = chromadb.Client()
COLLECTION_NAME = "incident_memory"


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def load_incident_rows():
    df = pd.read_csv(CSV_PATH)

    documents = []
    ids = []

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

    return documents, ids


def reset_memory():
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()

    documents, ids = load_incident_rows()

    collection.add(
        documents=documents,
        ids=ids
    )

    return collection.count()


def initialize_memory():
    collection = get_collection()

    if collection.count() == 0:
        documents, ids = load_incident_rows()
        collection.add(
            documents=documents,
            ids=ids
        )

    return collection.count()


def search_incidents(query, n_results=3):
    collection = get_collection()

    if collection.count() == 0:
        initialize_memory()

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    documents = results["documents"][0]

    return "\n\n--- Similar Incident ---\n\n".join(documents)


initialize_memory()