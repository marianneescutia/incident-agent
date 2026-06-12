import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.action_prompt import ACTION_OUTPUT_INSTRUCTIONS
from utils.config import (
    ACTION_ADAPTER_PATH,
    ACTION_BASE_MODEL,
    DEVICE_MAP,
    MODEL_DTYPE,
)
from utils.json_utils import require_json_fields

tokenizer = None
model = None


def load_grpo_action_model():
    global tokenizer
    global model

    if model is not None:
        return

    if not ACTION_ADAPTER_PATH.exists():
        raise FileNotFoundError(
            "GRPO adapter not found at "
            f"{ACTION_ADAPTER_PATH}. Set INCIDENT_ACTION_ADAPTER or run "
            "python finetuning/grpo_trainer.py."
        )

    tokenizer = AutoTokenizer.from_pretrained(ACTION_BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        ACTION_BASE_MODEL,
        dtype=MODEL_DTYPE,
        device_map=DEVICE_MAP,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(ACTION_ADAPTER_PATH),
    )

    model.eval()


def action_agent(
    log,
    analytics,
    prediction,
    retrieved_incident
):

    start = time.time()

    load_grpo_action_model()

    prompt = f"""
You are an enterprise remediation specialist.

Current Incident:
{log}

Analytics:
{analytics}

ML Prediction:
{prediction}

Similar Historical Incidents:
{retrieved_incident}

{ACTION_OUTPUT_INSTRUCTIONS}
"""

    messages = [
        {
            "role": "system",
            "content": "You are a GRPO fine-tuned remediation action agent."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    required_fields = [
        "immediate_action",
        "short_term_mitigation",
        "long_term_prevention",
    ]
    input_tokens = 0
    output_tokens = 0
    response = None

    for attempt in range(2):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(model.device)
        attempt_input_tokens = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        attempt_output_tokens = outputs[0].shape[0] - attempt_input_tokens
        raw_response = tokenizer.decode(
            outputs[0][attempt_input_tokens:],
            skip_special_tokens=True
        )
        input_tokens += attempt_input_tokens
        output_tokens += attempt_output_tokens

        try:
            response = require_json_fields(raw_response, required_fields)
            break
        except ValueError:
            messages.append({"role": "assistant", "content": raw_response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only valid JSON with exactly these fields: "
                        + ", ".join(required_fields)
                    ),
                }
            )

    if response is None:
        raise ValueError(
            "Action model failed to return valid structured JSON after 2 attempts."
        )

    latency = round(time.time() - start, 2)

    return {
        "response": response,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(output_tokens / max(latency, 0.001), 2),
        "attempts": attempt + 1,
        "model_type": "GRPO + LoRA Action Agent",
        "base_model": ACTION_BASE_MODEL,
        "adapter_path": str(ACTION_ADAPTER_PATH),
    }
