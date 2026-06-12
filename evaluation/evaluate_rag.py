import pandas as pd

from memory.vector_store import search_incident_records
from utils.config import PROJECT_ROOT, RAG_TOP_K, TEST_DATA_PATH, TRAIN_DATA_PATH


def main():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    rows = []

    for _, row in test_df.iterrows():
        results = search_incident_records(row["log"], n_results=RAG_TOP_K)
        metadatas = results["metadatas"][0]
        retrieved_categories = [item["category"] for item in metadatas]
        retrieved_risk_levels = [item["risk_level"] for item in metadatas]
        retrieved_severities = [item["severity"] for item in metadatas]
        category_has_train_example = bool(
            (train_df["category"] == row["category"]).any()
        )
        rows.append(
            {
                "incident_id": row["id"],
                "expected_category": row["category"],
                "retrieved_categories": "|".join(retrieved_categories),
                "category_has_train_example": category_has_train_example,
                f"category_recall_at_{RAG_TOP_K}": int(
                    row["category"] in retrieved_categories
                ),
                f"risk_recall_at_{RAG_TOP_K}": int(
                    row["risk_level"] in retrieved_risk_levels
                ),
                f"severity_recall_at_{RAG_TOP_K}": int(
                    row["severity"] in retrieved_severities
                ),
            }
        )

    detailed = pd.DataFrame(rows)
    category_metric = f"category_recall_at_{RAG_TOP_K}"
    evaluable_categories = detailed[detailed["category_has_train_example"]]
    summary = pd.DataFrame(
        [
            {
                "test_samples": len(detailed),
                "top_k": RAG_TOP_K,
                f"risk_recall_at_{RAG_TOP_K}": detailed[
                    f"risk_recall_at_{RAG_TOP_K}"
                ].mean(),
                f"severity_recall_at_{RAG_TOP_K}": detailed[
                    f"severity_recall_at_{RAG_TOP_K}"
                ].mean(),
                "category_evaluable_samples": len(evaluable_categories),
                f"category_recall_at_{RAG_TOP_K}_when_available": (
                    evaluable_categories[category_metric].mean()
                    if len(evaluable_categories)
                    else None
                ),
            }
        ]
    )
    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "rag_detailed.csv", index=False)
    summary.to_csv(output_dir / "rag_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
