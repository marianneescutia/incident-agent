from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "sample_data" / "incidents.csv"
TRAIN_PATH = PROJECT_ROOT / "sample_data" / "incidents_train.csv"
TEST_PATH = PROJECT_ROOT / "sample_data" / "incidents_test.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def create_splits():
    source = pd.read_csv(SOURCE_PATH)
    train, test = train_test_split(
        source,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=source["risk_level"],
    )

    train = train.sort_values("id").reset_index(drop=True)
    test = test.sort_values("id").reset_index(drop=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    overlap = set(train["id"]) & set(test["id"])
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {sorted(overlap)}")

    print(f"Created {len(train)} training rows at {TRAIN_PATH}")
    print(f"Created {len(test)} held-out rows at {TEST_PATH}")


if __name__ == "__main__":
    create_splits()
