import json
import re


def extract_json(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    cleaned = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def require_json_fields(text, required_fields):
    parsed = extract_json(text)
    if parsed is None:
        raise ValueError("Model response was not valid JSON.")

    missing = [field for field in required_fields if field not in parsed]
    if missing:
        raise ValueError(
            "Model response is missing required fields: " + ", ".join(missing)
        )
    return parsed
