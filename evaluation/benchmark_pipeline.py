import argparse
import json
import statistics
import time

from agents.coordinator import coordinator_agent
from utils.config import PROJECT_ROOT


DEFAULT_INCIDENT = """CRITICAL: Unauthorized access attempt
150 failed logins
Multiple IP addresses detected"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--incident", default=DEFAULT_INCIDENT)
    args = parser.parse_args()

    print("Warm-up run...")
    coordinator_agent(args.incident)

    runs = []
    for index in range(args.runs):
        start = time.perf_counter()
        result = coordinator_agent(args.incident)
        wall_time = time.perf_counter() - start
        runs.append(
            {
                "run": index + 1,
                "wall_time_seconds": round(wall_time, 4),
                **result["metrics"],
            }
        )

    summary = {
        "measured_runs": args.runs,
        "mean_wall_time_seconds": round(
            statistics.mean(run["wall_time_seconds"] for run in runs),
            4,
        ),
        "median_wall_time_seconds": round(
            statistics.median(run["wall_time_seconds"] for run in runs),
            4,
        ),
        "mean_action_tokens_per_second": round(
            statistics.mean(run["action_tokens_per_second"] for run in runs),
            2,
        ),
        "peak_gpu_memory_gb": max(run["gpu_peak_memory_gb"] for run in runs),
        "gpu_name": runs[0]["gpu_name"],
        "rocm_version": runs[0]["rocm_version"],
    }

    payload = {"summary": summary, "runs": runs}
    output_dir = PROJECT_ROOT / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pipeline_benchmark.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
