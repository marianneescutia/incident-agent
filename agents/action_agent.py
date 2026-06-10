import os
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "/workspace/shared/models/action-agent-grpo"

tokenizer = None
model = None


def load_grpo_action_model():
    global tokenizer
    global model

    if model is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype="auto",
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
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

Return ONLY valid JSON:

{{
  "immediate_action": "",
  "short_term_mitigation": "",
  "long_term_prevention": ""
}}
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

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    output_tokens = outputs[0].shape[0] - input_tokens

    response = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True
    )

    response = response.replace("```json", "").replace("```", "").strip()

    latency = round(time.time() - start, 2)

    return {
        "response": response,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_type": "GRPO + LoRA Action Agent"
    }
