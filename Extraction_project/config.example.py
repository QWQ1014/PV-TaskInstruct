"""Copy to config.py and provide credentials via environment variables."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = os.getenv("PV_SOURCE_DIRECTORY", str(PROJECT_ROOT / "Extraction_project" / "data" / "source_papers"))
FILE_PATTERN = "**/*.md"

_settings = {
    "api_key": os.getenv("PV_API_KEY", ""),
    "base_url": os.getenv("PV_API_BASE_URL", "https://api.siliconflow.cn/v1"),
    "model": os.getenv("PV_MODEL_NAME", ""),
    "timeout": int(os.getenv("PV_API_TIMEOUT", "500")),
    "concurrency": int(os.getenv("PV_API_CONCURRENCY", "3")),
    "rpm_limit": int(os.getenv("PV_API_RPM_LIMIT", "60")),
}

API_CONFIG = {
    alias: dict(_settings)
    for alias in (
        "silicon",
        "silicon_deepseek-v3.2",
        "silicon_deepseek_r1_0528",
        "silicon_01",
        "silicon_glm_z1_01",
        "silicon_deepseek_r1_0528_01",
        "silicon_glm_4",
    )
}
