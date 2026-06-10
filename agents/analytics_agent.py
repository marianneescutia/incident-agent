from utils.amd_llm import ask_llm


def analytics_agent(log):
    prompt = f"""
Analyze this incident:

{log}

Return ONLY valid JSON:

{{
  "category": "",
  "severity": "",
  "impact": ""
}}
"""

    return ask_llm(
        prompt,
        system="You are an enterprise incident analyst.",
        max_tokens=160
    )
