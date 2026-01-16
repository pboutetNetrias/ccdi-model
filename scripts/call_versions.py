#!/usr/bin/env python3
import json
import os
import time
from typing import Any, Dict, Optional

import requests

BASE_URL = os.getenv("BASE_URL", "https://85fnwlcuc2.execute-api.us-east-2.amazonaws.com/default")
API_KEY = os.getenv("API_KEY")
DATA_COMMONS_KEY = os.getenv("DATA_COMMONS_KEY", "ccdi").strip()
PAYLOAD_PATH = os.getenv("PAYLOAD_PATH", "payloads/version_payload.json")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))

def call_versions(
    data_commons_key: str,
    payload: Dict[str, Any],
    query: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    url = f"{BASE_URL}/data-models/{data_commons_key}/versions"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY

    t0 = time.time()
    resp = requests.post(url, params=(query or {}), json=payload, headers=headers, timeout=timeout)
    elapsed = round(time.time() - t0, 3)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP {resp.status_code}: {e}\nResponse text:\n{resp.text}")
        raise

    data = resp.json()
    if isinstance(data, dict) and "body" in data:
        body_json = data["body"]
        data = json.loads(body_json) if isinstance(body_json, str) else body_json

    data["_latency_s"] = elapsed
    return data

def make_unique_version_label() -> str:
    # Most unique + readable: auto-v<run>-<sha>
    run = os.getenv("GITHUB_RUN_NUMBER", "0")
    sha = os.getenv("GITHUB_SHA", "")[:7] or "nosha"
    prefix = os.getenv("VERSION_PREFIX", "auto-")  # optional
    return f"{prefix}v{run}-{sha}"

def main():
    with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Always ensure uniqueness unless explicitly overridden
    payload["version_label"] = os.getenv("VERSION_LABEL") or make_unique_version_label()

    result = call_versions(DATA_COMMONS_KEY, payload, timeout=TIMEOUT)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
