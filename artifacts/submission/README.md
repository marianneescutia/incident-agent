# Final Submission Evidence

These compact artifacts record the final measurements produced in the AMD
Jupyter environment on June 12, 2026.

- `amd_environment.json`: Python, PyTorch, ROCm, and MI300X device evidence.
- `prediction_summary.csv`: held-out risk-classification metrics.
- `rag_summary.csv`: held-out retrieval metrics.
- `action_model_summary.csv`: base-vs-GRPO action-model comparison.
- `pipeline_benchmark.json`: warm end-to-end MI300X benchmark.

The benchmark utility reported the generic label `AMD Instinct GPU`; the system
report identified the device specifically as an Instinct MI300X. Generated
runtime artifacts remain ignored so rerunning evaluation does not create
repository noise.
