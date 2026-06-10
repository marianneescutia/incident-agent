import time
import torch

from agents.analytics_agent import analytics_agent
from agents.prediction_agent import prediction_agent
from agents.action_agent import action_agent
from agents.report_agent import report_agent

from memory.vector_store import search_incidents


def coordinator_agent(log):
    workflow_start = time.time()

    retrieved_incident = search_incidents(
        log,
        n_results=3
    )

    analytics = analytics_agent(log)

    prediction = prediction_agent(log)

    actions = action_agent(
        log,
        analytics["response"],
        prediction["response"],
        retrieved_incident
    )

    report = report_agent(
        log,
        analytics["response"],
        prediction["response"],
        actions["response"]
    )

    total_time = round(
        time.time() - workflow_start,
        2
    )

    try:
        gpu_memory = round(
            torch.cuda.memory_allocated() / 1024**3,
            2
        )

        gpu_reserved = round(
            torch.cuda.memory_reserved() / 1024**3,
            2
        )

    except Exception:
        gpu_memory = 0
        gpu_reserved = 0

    metrics = {
        "analytics_latency": analytics["latency"],
        "prediction_latency": prediction["latency"],
        "action_latency": actions["latency"],
        "report_latency": report["latency"],
        "total_latency": total_time,

        "analytics_tokens": analytics["output_tokens"],
        "prediction_tokens": prediction["output_tokens"],
        "action_tokens": actions["output_tokens"],
        "report_tokens": report["output_tokens"],

        "gpu_memory_gb": gpu_memory,
        "gpu_reserved_gb": gpu_reserved,

        "prediction_model": "RandomForestClassifier + IsolationForest",
        "action_model": "GRPO + LoRA Action Agent",
        "rag_top_k": 3
    }

    return {
        "retrieved_incident": retrieved_incident,
        "analytics": analytics,
        "prediction": prediction,
        "actions": actions,
        "report": report,
        "metrics": metrics
    }
