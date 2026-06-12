import json
import re

from utils.action_prompt import ACTION_FIELDS


def extract_json(text):
    text = str(text).replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def json_format_reward(completions, **kwargs):
    rewards = []

    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        parsed = extract_json(content)

        if parsed is None:
            rewards.append(0.0)
            continue

        required = [
            "immediate_action",
            "short_term_mitigation",
            "long_term_prevention"
        ]

        field_score = sum(key in parsed for key in required) / len(required)

        # Stronger reward if output is exactly the desired JSON structure
        extra_keys = set(parsed.keys()) - set(required)
        structure_bonus = 0.25 if len(extra_keys) == 0 else 0.0

        rewards.append(field_score + structure_bonus)

    return rewards


def remediation_keyword_reward(completions, **kwargs):
    expected_immediate = kwargs.get("expected_immediate_action", [])
    expected_short = kwargs.get("expected_short_term_mitigation", [])
    expected_long = kwargs.get("expected_long_term_prevention", [])

    rewards = []

    for i, completion in enumerate(completions):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        content_lower = str(content).lower()

        expected_text = " ".join([
            str(expected_immediate[i]),
            str(expected_short[i]),
            str(expected_long[i])
        ]).lower()

        expected_words = [
            word for word in re.findall(r"[a-zA-Z]+", expected_text)
            if len(word) > 4
        ]

        if not expected_words:
            rewards.append(0.0)
            continue

        matches = sum(
            1 for word in expected_words
            if word in content_lower
        )

        base_score = matches / len(expected_words)

        # Weight keyword alignment more strongly
        rewards.append(base_score * 3.0)

    return rewards


def safety_reward(completions, **kwargs):
    unsafe_terms = [
        "do nothing",
        "ignore",
        "delete all",
        "disable security",
        "share password",
        "turn off monitoring"
    ]

    rewards = []

    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        content_lower = str(content).lower()

        if any(term in content_lower for term in unsafe_terms):
            rewards.append(0.0)
        else:
            rewards.append(0.5)

    return rewards


def concise_json_reward(completions, **kwargs):
    rewards = []

    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        raw = str(content).strip()
        parsed = extract_json(raw)

        if parsed is None or set(parsed.keys()) != set(ACTION_FIELDS):
            rewards.append(0.0)
            continue

        values = [parsed[field] for field in ACTION_FIELDS]
        if not all(isinstance(value, str) and value.strip() for value in values):
            rewards.append(0.0)
            continue

        word_counts = [len(re.findall(r"[A-Za-z0-9]+", value)) for value in values]
        concise_score = sum(count <= 12 for count in word_counts) / len(word_counts)

        cleaned = raw.replace("```json", "").replace("```", "").strip()
        object_match = re.fullmatch(r"\{.*\}", cleaned, re.DOTALL)
        exact_object_bonus = 0.5 if object_match else 0.0
        rewards.append(concise_score + exact_object_bonus)

    return rewards
