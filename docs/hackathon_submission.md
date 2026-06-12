# Hackathon Submission Content

Use this file as the source for the required 1-5 slide presentation.

## Slide 1: Problem and relevance

**Title:** Incident Intelligence Agent

Enterprise responders must turn fragmented logs into risk decisions and safe
remediation under severe time pressure. Manual triage is slow, inconsistent,
and difficult to communicate to business stakeholders.

**Target users:** SRE, IT operations, SOC analysts, incident commanders, and
service owners.

**Value:** reduce time-to-understand, make prior resolutions searchable, and
produce a consistent action plan plus executive report.

## Slide 2: Solution

Show the diagram from `docs/architecture.md`.

The workflow combines:

- Phi-3 analytics and executive reporting.
- Random Forest risk classification and Isolation Forest anomaly detection.
- Lightweight TF-IDF retrieval over resolved incidents.
- Qwen2.5 GRPO + LoRA remediation.
- Streamlit evidence and performance dashboard.

## Slide 3: What was built

- Five coordinated analysis stages and a working UI.
- Structured JSON validation and explicit failure handling.
- Reproducible train/test isolation.
- GRPO reward functions for structure, remediation relevance, and safety.
- AMD environment and benchmark capture.

**Dataset:** 50 synthetic enterprise incident scenarios. Forty are used for
development and ten are held out. State clearly that this is prototype data.

## Slide 4: Technical results

Run the evaluation scripts on the final AMD environment and paste:

- Prediction accuracy and macro-F1.
- Retrieval risk/severity Recall@3 and category recall where represented.
- Base vs GRPO action token F1 and JSON validity.
- End-to-end warm latency.
- Action-agent tokens per second.
- Peak MI300X memory.
- ROCm and PyTorch versions.

Never use the deleted legacy `eval_results.csv`; it did not represent model
inference.

## Slide 5: Impact, innovation, and future work

**Differentiator:** one traceable workflow combines retrieval, interpretable ML,
reinforcement fine-tuning, and executive communication instead of relying on a
single generic chatbot.

**Expected impact:** faster triage, more consistent actions, reuse of historical
knowledge, and clearer incident communication.

**Limitations:** small synthetic dataset, lexical safety checks, no production
tool execution, and no human-approval workflow.

**Future work:** sanitized enterprise data, larger independent test set,
persistent vector storage, RBAC, secret redaction, tool execution with approval
gates, and concurrent serving benchmarks.

Add the final demo video URL and GitHub URL:
https://github.com/marianneescutia/incident-agent

## Demo script

1. Show `amd-smi` and `artifacts/system/amd_environment.json`.
2. Open Streamlit and select a security incident.
3. Explain the three retrieved incidents.
4. Show risk, confidence, and anomaly output.
5. Show the GRPO remediation JSON.
6. Show the executive report.
7. End on the held-out results and MI300X benchmark.

Keep the recording between two and three minutes and record one backup take.
