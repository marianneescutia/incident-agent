import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.getenv("INCIDENT_DATA_DIR", PROJECT_ROOT / "sample_data"))
TRAIN_DATA_PATH = Path(
    os.getenv("INCIDENT_TRAIN_DATA", DATA_DIR / "incidents_train.csv")
)
TEST_DATA_PATH = Path(
    os.getenv("INCIDENT_TEST_DATA", DATA_DIR / "incidents_test.csv")
)

LLM_MODEL = os.getenv(
    "INCIDENT_LLM_MODEL",
    "microsoft/Phi-3-mini-4k-instruct",
)
ACTION_BASE_MODEL = os.getenv(
    "INCIDENT_ACTION_BASE_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
ACTION_ADAPTER_PATH = Path(
    os.getenv(
        "INCIDENT_ACTION_ADAPTER",
        PROJECT_ROOT / "models" / "action-agent-grpo",
    )
)

RAG_TOP_K = int(os.getenv("INCIDENT_RAG_TOP_K", "3"))
MODEL_DTYPE = os.getenv("INCIDENT_MODEL_DTYPE", "auto")
DEVICE_MAP = os.getenv("INCIDENT_DEVICE_MAP", "auto")
