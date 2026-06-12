import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from agents.prediction_agent import prediction_agent
from utils.config import PROJECT_ROOT, TEST_DATA_PATH


def main():
    test_df = pd.read_csv(TEST_DATA_PATH)
    expected = test_df["risk_level"].tolist()
    predicted = [
        prediction_agent(log)["response"]["risk_level"]
        for log in test_df["log"]
    ]

    summary = pd.DataFrame(
        [
            {
                "test_samples": len(test_df),
                "accuracy": accuracy_score(expected, predicted),
                "macro_f1": f1_score(
                    expected,
                    predicted,
                    average="macro",
                    zero_division=0,
                ),
            }
        ]
    )
    labels = sorted(set(expected) | set(predicted))
    matrix = pd.DataFrame(
        confusion_matrix(expected, predicted, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    report = pd.DataFrame(
        classification_report(
            expected,
            predicted,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    matrix.to_csv(output_dir / "prediction_confusion_matrix.csv")
    report.to_csv(output_dir / "prediction_classification_report.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
