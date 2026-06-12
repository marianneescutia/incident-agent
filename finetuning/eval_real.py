import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetuning.eval import (
    action_token_f1,
    required_fields_score,
    safety_score,
    valid_json_score,
)
from utils.config import (
    ACTION_ADAPTER_PATH,
    ACTION_BASE_MODEL,
    DEVICE_MAP,
    MODEL_DTYPE,
    PROJECT_ROOT,
    TEST_DATA_PATH,
)


def build_prompt(row):
    return f"""
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


def generate(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "You are an enterprise remediation specialist."},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    output_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return response, latency, output_tokens


def evaluate_model(model_name, model, tokenizer, test_df):
    rows = []
    for _, row in test_df.iterrows():
        output, latency, output_tokens = generate(
            model,
            tokenizer,
            build_prompt(row),
        )
        expected_text = " ".join(
            [
                str(row["immediate_action"]),
                str(row["short_term_mitigation"]),
                str(row["long_term_prevention"]),
            ]
        )
        rows.append(
            {
                "incident_id": row["id"],
                "model": model_name,
                "valid_json": valid_json_score(output),
                "required_fields": required_fields_score(output),
                "action_token_f1": action_token_f1(output, expected_text),
                "safety": safety_score(output),
                "latency_seconds": round(latency, 4),
                "output_tokens": output_tokens,
                "tokens_per_second": round(
                    output_tokens / max(latency, 0.001),
                    2,
                ),
                "sample_output": output,
            }
        )
    return rows


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        ACTION_BASE_MODEL,
        dtype=MODEL_DTYPE,
        device_map=DEVICE_MAP,
    )


def main():
    if not ACTION_ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Adapter not found at {ACTION_ADAPTER_PATH}. "
            "Train it first or set INCIDENT_ACTION_ADAPTER."
        )

    test_df = pd.read_csv(TEST_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(ACTION_BASE_MODEL)

    print(f"Evaluating {len(test_df)} held-out incidents.")
    base_model = load_base_model()
    base_rows = evaluate_model("base", base_model, tokenizer, test_df)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    grpo_model = PeftModel.from_pretrained(
        load_base_model(),
        str(ACTION_ADAPTER_PATH),
    )
    grpo_model.eval()
    grpo_rows = evaluate_model("grpo", grpo_model, tokenizer, test_df)

    results = pd.DataFrame(base_rows + grpo_rows)
    metric_columns = [
        "valid_json",
        "required_fields",
        "action_token_f1",
        "safety",
        "latency_seconds",
        "tokens_per_second",
    ]
    summary = results.groupby("model")[metric_columns].mean().round(3)

    output_dir = PROJECT_ROOT / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "action_model_detailed.csv", index=False)
    summary.to_csv(output_dir / "action_model_summary.csv")

    print(summary)
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
