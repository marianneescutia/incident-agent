# AMD MI300X Runbook

## Before using GPU time

Keep the repository, model adapter, benchmark outputs, and any downloaded model
cache under `/workspace/shared` so they survive a stopped session.

```bash
cd /workspace/shared/incident-agent
set -a
source .env
set +a
bash scripts/setup_amd.sh
python -m unittest discover -s tests -v
python -m scripts.system_report
```

Verify that the report contains:

- `"cuda_api_available": true`
- a non-null `"rocm_version"`
- an AMD Instinct MI300X device

## Training

```bash
export INCIDENT_ACTION_ADAPTER=/workspace/shared/models/action-agent-grpo
python -m finetuning.grpo_trainer
```

Save the terminal log. Record training duration, max steps, adapter size, and
peak memory for the presentation.

TRL is pinned to `0.24.0`, which matches the preinstalled Python 3.12,
Transformers 4.57, PEFT 0.17, Accelerate 1.10, and Datasets 4.2 stack.

## Evaluation

```bash
python -m evaluation.evaluate_prediction
python -m evaluation.evaluate_rag
python -m finetuning.eval_real
python -m evaluation.benchmark_pipeline --runs 3
```

Do not report cold-start latency as steady-state latency. The benchmark performs
one warm-up before measured runs.

## Demo

```bash
streamlit run app.py
```

Run the exact incident planned for the video before recording. Keep a screenshot
or recording of a successful result in case notebook capacity is unavailable.
