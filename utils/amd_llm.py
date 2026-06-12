import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import DEVICE_MAP, LLM_MODEL, MODEL_DTYPE
from utils.json_utils import require_json_fields

tokenizer = None
model = None


def load_llm():
    global tokenizer
    global model

    if model is not None:
        return

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        dtype=MODEL_DTYPE,
        device_map=DEVICE_MAP,
    )
    model.eval()


def clean_response(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return text.strip()


def ask_llm(
    prompt,
    system="You are an enterprise incident assistant.",
    max_tokens=200
):
    load_llm()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
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

    start_time = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = round(
        time.time() - start_time,
        2
    )

    output_tokens = outputs[0].shape[0] - input_tokens

    response = tokenizer.decode(
        outputs[0][input_tokens:],
        skip_special_tokens=True
    )

    response = clean_response(response)

    return {
        "response": response,
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(output_tokens / max(latency, 0.001), 2),
        "model_type": LLM_MODEL,
    }


def ask_llm_json(prompt, required_fields, system, max_tokens=200):
    current_prompt = prompt
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(2):
        result = ask_llm(
            current_prompt,
            system=system,
            max_tokens=max_tokens,
        )
        total_latency += result["latency"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        try:
            result["response"] = require_json_fields(
                result["response"],
                required_fields,
            )
            result["latency"] = round(total_latency, 2)
            result["input_tokens"] = total_input_tokens
            result["output_tokens"] = total_output_tokens
            result["tokens_per_second"] = round(
                total_output_tokens / max(total_latency, 0.001),
                2,
            )
            result["attempts"] = attempt + 1
            return result
        except ValueError:
            current_prompt = (
                f"{prompt}\n\nYour previous response was invalid. "
                "Return only one valid JSON object with exactly these fields: "
                + ", ".join(required_fields)
            )

    raise ValueError("Model failed to return valid structured JSON after 2 attempts.")
