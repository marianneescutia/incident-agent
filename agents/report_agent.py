from utils.amd_llm import ask_llm


def report_agent(
    log,
    analytics,
    prediction,
    actions
):

    prompt = f"""
Incident:
{log}

Analytics:
{analytics}

ML Prediction:
{prediction}

Actions:
{actions}

Generate an executive incident report.

Include:

- Summary
- Business Impact
- Risk Assessment
- Recommended Actions
- Next Steps

Keep it concise and professional.
"""

    return ask_llm(
        prompt,
        system="You are an executive reporting assistant.",
        max_tokens=300
    )
