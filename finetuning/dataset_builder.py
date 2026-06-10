import os
import pandas as pd
from datasets import Dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "sample_data", "incidents.csv")


def build_action_dataset():
    df = pd.read_csv(CSV_PATH)

    rows = []

    for _, row in df.iterrows():

        prompt = f"""
You are an enterprise remediation specialist.

Incident:
{row["log"]}

Category:
{row["category"]}

Severity:
{row["severity"]}

Risk Level:
{row["risk_level"]}

Return ONLY valid JSON:

{{
  "immediate_action": "",
  "short_term_mitigation": "",
  "long_term_prevention": ""
}}
"""

        expected = {
            "immediate_action": row["immediate_action"],
            "short_term_mitigation": row["short_term_mitigation"],
            "long_term_prevention": row["long_term_prevention"]
        }

        rows.append(
            {
                "prompt": prompt,
                "expected_immediate_action": expected["immediate_action"],
                "expected_short_term_mitigation": expected["short_term_mitigation"],
                "expected_long_term_prevention": expected["long_term_prevention"]
            }
        )

    return Dataset.from_list(rows)


if __name__ == "__main__":
    dataset = build_action_dataset()
    print(dataset)
    print(dataset[0])
