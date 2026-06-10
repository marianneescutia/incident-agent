from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

print("Loading Phi-3...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto"
)

print("Phi-3 ready")


def clean_response(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return text.strip()


def ask_llm(
    prompt,
    system="You are an enterprise incident assistant.",
    max_tokens=200
):

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

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
        "output_tokens": output_tokens
    }