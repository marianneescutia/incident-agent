# Incident Intelligence Agent

An enterprise incident-response prototype built for the AMD TCS Hackathon. It
combines a multi-agent LLM workflow, TF-IDF retrieval over historical incidents,
classical risk/anomaly models, and a GRPO + LoRA remediation agent.

## Problem

Incident responders must correlate noisy logs, estimate operational risk, find
similar incidents, recommend safe actions, and communicate impact under time
pressure. This project turns a raw incident log into:

1. Structured incident analysis.
2. Risk and anomaly prediction.
3. Retrieval of similar resolved incidents.
4. Immediate, short-term, and long-term remediation actions.
5. An executive incident report.

## Architecture

```text
Incident log
    |
    +--> TF-IDF retrieval (training corpus only)
    +--> Phi-3 Analytics Agent
    +--> RandomForest + IsolationForest Prediction Agent
              |
              v
       Qwen GRPO + LoRA Action Agent
              |
              v
         Phi-3 Report Agent
              |
              v
        Streamlit UI + metrics
```

## Data isolation

`sample_data/incidents.csv` is deterministically split into:

- `incidents_train.csv`: 40 records used by GRPO, Random Forest,
  Isolation Forest, and the RAG corpus.
- `incidents_test.csv`: 10 held-out records used only by evaluation.

Regenerate and verify the split with:

```bash
python -m scripts.create_data_splits
python -m unittest discover -s tests
```

## AMD MI300X / ROCm setup

This project is configured for the observed AMD environment: Python 3.12,
PyTorch 2.8 ROCm 7.0, Transformers 4.57, PEFT 0.17, Accelerate 1.10, and one
MI300X with approximately 192 GB HBM. Do not replace its preinstalled `torch`
package with a CUDA or PyPI wheel.

```bash
cd /workspace/shared
git clone https://github.com/marianneescutia/incident-agent.git
cd incident-agent
cp .env.example .env
set -a
source .env
set +a
bash scripts/setup_amd.sh
```

Export the variables for the notebook shell:

```bash
export INCIDENT_ACTION_ADAPTER=/workspace/shared/models/action-agent-grpo
export INCIDENT_TRAIN_DATA=/workspace/shared/incident-agent/sample_data/incidents_train.csv
export INCIDENT_TEST_DATA=/workspace/shared/incident-agent/sample_data/incidents_test.csv
```

Confirm that PyTorch sees ROCm and the MI300X:

```bash
python -m scripts.system_report
amd-smi
```

The report is saved to `artifacts/system/amd_environment.json`. The setup script
installs only the packages missing from the AMD image: TRL 0.24, Streamlit, and
SentencePiece.

## Train the remediation adapter

```bash
python -m finetuning.grpo_trainer
```

The default output is `models/action-agent-grpo`. Set
`INCIDENT_ACTION_ADAPTER` to store it in persistent `/workspace/shared`
storage.

## Run the application

```bash
streamlit run app.py
```

The first request is a cold start and includes model loading. Use later requests
when reporting steady-state latency.

## Held-out evaluation

```bash
python -m evaluation.evaluate_prediction
python -m evaluation.evaluate_rag
python -m finetuning.eval_real
```

These commands save presentation-ready CSV files under
`artifacts/evaluation/`:

- Risk classification accuracy, macro-F1, and confusion matrix.
- Retrieval risk/severity Recall@3 and category recall where the training corpus
  contains that category.
- Base-vs-GRPO JSON validity, field completeness, action token F1, safety,
  latency, and tokens per second.

The test set is never loaded by training or retrieval code.

## MI300X benchmark

Run one warm-up followed by measured steady-state executions:

```bash
python -m evaluation.benchmark_pipeline --runs 3
```

The JSON artifact includes end-to-end latency, tokens per second, peak GPU
memory, GPU name, and ROCm version.

## Demo flow

1. Select a database, security, GPU, or RAG incident.
2. Show the retrieved historical incidents.
3. Show risk, anomaly, and confidence.
4. Show the GRPO remediation plan.
5. Show the executive report.
6. Close with held-out quality metrics and MI300X performance.

## Limitations and future work

- The current dataset is small and synthetic; production validation requires
  sanitized real incident histories.
- Safety evaluation is a first-pass lexical check, not a substitute for human
  approval or policy enforcement.
- Future work includes a production vector database, larger independent test
  sets, tool-executing agents with approval gates, secret redaction, RBAC, and
  concurrent inference benchmarks.

## Repository structure

- `agents/`: analytics, prediction, remediation, report, and coordination.
- `memory/`: training-only incident retrieval.
- `finetuning/`: GRPO dataset, rewards, training, and held-out evaluation.
- `evaluation/`: ML, RAG, and MI300X pipeline benchmarks.
- `scripts/`: split generation and environment evidence.
- `tests/`: data leakage and JSON validation checks.
- `docs/`: architecture, AMD runbook, slide content, and demo script.
