import re


def extract_number(pattern, text, default=0):
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def extract_features(log):
    text = log.lower()

    cpu = extract_number(r"cpu\s*(\d+)", text)
    memory = extract_number(r"memory\s*(\d+)", text)
    failed_logins = extract_number(r"(\d+)\s*failed logins", text)
    packet_loss = extract_number(r"packet loss\s*(\d+)", text)
    error_500 = 1 if "500" in text or "503" in text else 0

    security_keywords = [
        "unauthorized",
        "failed logins",
        "ransomware",
        "privilege",
        "secrets",
        "breach",
        "exfiltration",
        "prompt injection"
    ]

    database_keywords = [
        "database",
        "replication",
        "deadlock",
        "connection"
    ]

    infra_keywords = [
        "disk",
        "pod",
        "kubernetes",
        "dns",
        "load balancer",
        "storage"
    ]

    ai_keywords = [
        "model",
        "inference",
        "embedding",
        "token",
        "vector"
    ]

    features = {
        "cpu": cpu,
        "memory": memory,
        "failed_logins": failed_logins,
        "packet_loss": packet_loss,
        "error_500": error_500,
        "has_security_keyword": int(any(k in text for k in security_keywords)),
        "has_database_keyword": int(any(k in text for k in database_keywords)),
        "has_infra_keyword": int(any(k in text for k in infra_keywords)),
        "has_ai_keyword": int(any(k in text for k in ai_keywords)),
        "has_critical": int("critical" in text),
        "has_error": int("error" in text),
        "has_warning": int("warning" in text),
        "log_length": len(text)
    }

    return features
