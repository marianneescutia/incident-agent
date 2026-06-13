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

Use the final measured submission results:

- Prediction: 0.50 accuracy and 0.417 macro-F1 on 10 held-out incidents.
- Retrieval: 0.90 risk Recall@3 and 0.90 severity Recall@3.
- Exact category Recall@3: 0.20 on the five held-out categories represented in
  the training corpus.
- Action model: valid JSON 1.00 and required fields 1.00 for base and GRPO.
- Action-token F1: 0.132 base vs 0.137 GRPO, a 3.8% relative improvement.
- Action latency: 0.886 seconds base vs 1.274 seconds GRPO.
- Warm pipeline benchmark: 8.16 seconds mean, 43.29 action tokens/second, and
  8.46 GB peak GPU memory.
- Environment: AMD Instinct MI300X, ROCm 7.0, PyTorch 2.8, Python 3.12.

Frame the fine-tuning result as a modest quality improvement with a latency
trade-off. Do not claim a throughput improvement.

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
