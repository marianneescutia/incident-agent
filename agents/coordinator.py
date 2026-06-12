import time
import os
import torch

from agents.analytics_agent import analytics_agent
from agents.prediction_agent import prediction_agent
from agents.action_agent import action_agent
from agents.report_agent import report_agent

from memory.vector_store import search_incidents
from utils.config import RAG_TOP_K


def gpu_metrics():
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_name": "CPU",
            "rocm_version": None,
            "gpu_memory_allocated_gb": 0,
            "gpu_memory_reserved_gb": 0,
            "gpu_peak_memory_gb": 0,
        }

    return {
        "gpu_available": True,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            or os.getenv("AMD_GPU_NAME", "AMD Instinct GPU")
        ),
        "rocm_version": getattr(torch.version, "hip", None),
        "gpu_memory_allocated_gb": round(
            torch.cuda.memory_allocated() / 1024**3, 2
        ),
        "gpu_memory_reserved_gb": round(
            torch.cuda.memory_reserved() / 1024**3, 2
        ),
        "gpu_peak_memory_gb": round(
            torch.cuda.max_memory_allocated() / 1024**3, 2
        ),
    }


def coordinator_agent(log):
    workflow_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    retrieved_incident = search_incidents(
        log,
        n_results=RAG_TOP_K,
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
        "analytics_tokens_per_second": analytics["tokens_per_second"],
        "action_tokens_per_second": actions["tokens_per_second"],
        "report_tokens_per_second": report["tokens_per_second"],

        "prediction_model": "RandomForestClassifier + IsolationForest",
        "action_model": "GRPO + LoRA Action Agent",
        "rag_top_k": RAG_TOP_K,
        **gpu_metrics(),
    }

    return {
        "retrieved_incident": retrieved_incident,
        "analytics": analytics,
        "prediction": prediction,
        "actions": actions,
        "report": report,
        "metrics": metrics
    }
