ACTION_FIELDS = [
    "immediate_action",
    "short_term_mitigation",
    "long_term_prevention",
]

ACTION_OUTPUT_INSTRUCTIONS = """
Return exactly one JSON object and nothing else.
Use exactly these three keys:
- immediate_action
- short_term_mitigation
- long_term_prevention

Each value must be one concise sentence of at most 12 words.
Do not use Markdown, code fences, explanations, lists, or extra keys.
End immediately after the closing brace.

Example shape:
{
  "immediate_action": "Isolate the affected service and preserve diagnostic evidence.",
  "short_term_mitigation": "Route traffic to a healthy replica while monitoring recovery.",
  "long_term_prevention": "Add automated failover tests and capacity alerts."
}
""".strip()

