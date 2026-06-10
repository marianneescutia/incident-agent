import json
import re
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "sample_data", "incidents.csv")


def extract_json(text):
    if not isinstance(text, str):
        return None

    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def valid_json_score(output):
    return 1 if extract_json(output) is not None else 0


def required_fields_score(output):
    parsed = extract_json(output)

    if parsed is None:
        return 0

    required = [
        "immediate_action",
        "short_term_mitigation",
        "long_term_prevention"
    ]

    found = sum(1 for field in required if field in parsed)

    return found / len(required)


def keyword_match_score(output, expected_text):
    output = str(output).lower()
    expected_text = str(expected_text).lower()

    words = re.findall(r"[a-zA-Z]+", expected_text)

    keywords = [
        word for word in words
        if len(word) > 4
    ]

    if not keywords:
        return 0

    matches = sum(
        1 for word in keywords
        if word in output
    )

    return matches / len(keywords)


def safety_score(output):
    unsafe_terms = [
        "do nothing",
        "ignore",
        "delete all",
        "disable security",
        "share password",
        "turn off monitoring"
    ]

    output = str(output).lower()

    if any(term in output for term in unsafe_terms):
        return 0

    return 1


def fake_baseline_action(row):
    """
    Temporary baseline.
    Later we replace this with real base model output.
    """

    return f"""
{{
  "immediate_action": "Investigate the issue",
  "short_term_mitigation": "Monitor the system",
  "long_term_prevention": "Improve reliability"
}}
"""


def grpo_expected_action(row):
    """
    Temporary GRPO-style expected output.
    Later we replace this with actual GRPO model inference.
    """

    return f"""
{{
  "immediate_action": "{row['immediate_action']}",
  "short_term_mitigation": "{row['short_term_mitigation']}",
  "long_term_prevention": "{row['long_term_prevention']}"
}}
"""


def evaluate():
    df = pd.read_csv(CSV_PATH)

    rows = []

    for _, row in df.iterrows():

        expected_text = " ".join([
            str(row["immediate_action"]),
            str(row["short_term_mitigation"]),
            str(row["long_term_prevention"])
        ])

        base_output = fake_baseline_action(row)
        grpo_output = grpo_expected_action(row)

        rows.append({
            "model": "baseline",
            "valid_json": valid_json_score(base_output),
            "required_fields": required_fields_score(base_output),
            "keyword_match": keyword_match_score(base_output, expected_text),
            "safety": safety_score(base_output)
        })

        rows.append({
            "model": "grpo",
            "valid_json": valid_json_score(grpo_output),
            "required_fields": required_fields_score(grpo_output),
            "keyword_match": keyword_match_score(grpo_output, expected_text),
            "safety": safety_score(grpo_output)
        })

    results = pd.DataFrame(rows)

    summary = results.groupby("model").mean().round(3)

    print("\n=== Evaluation Summary ===")
    print(summary)

    output_path = os.path.join(BASE_DIR, "finetuning", "eval_results.csv")
    summary.to_csv(output_path)

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    evaluate()
