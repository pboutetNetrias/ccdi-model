#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
import yaml

BASE_URL_DEFAULT = "https://85fnwlcuc2.execute-api.us-east-2.amazonaws.com/default"
DATA_COMMONS_KEY_DEFAULT = "ccdi"


# ----------------------------
# YAML -> bdi-kit schema helpers
# ----------------------------
def _strings_from_iterable(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for v in values:
        if isinstance(v, (str, int, float, bool)):
            out.append(str(v))
        elif isinstance(v, dict):
            for k in ("Value", "value", "label", "name", "preferred_label"):
                if k in v and isinstance(v[k], str):
                    out.append(v[k])
                    break
    return out


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_ccdi_yaml_to_bdi_model(yaml_path: Path) -> Dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as f:
        input_schema = yaml.safe_load(f) or {}

    prop_definitions = (input_schema.get("PropDefinitions") or {})
    if not isinstance(prop_definitions, dict):
        raise ValueError("PropDefinitions missing or not a dict in YAML")

    data_model: Dict[str, Any] = {}

    for cde, details in prop_definitions.items():
        if not isinstance(details, dict):
            continue

        data_model[cde] = {}
        data_model[cde]["column_description"] = details.get("Desc", "") or ""

        enums = _strings_from_iterable(details.get("Enum", []))

        item_type_vals: List[str] = []
        type_block = details.get("Type", {})
        if isinstance(type_block, dict):
            item_type_vals = _strings_from_iterable(type_block.get("item_type", []))

        combined = _unique_preserve_order(enums + item_type_vals)

        data_model[cde]["value_data"] = {pv: "" for pv in combined}

    return data_model


# ----------------------------
# Local state + git info (read-only)
# ----------------------------
def run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        # Don't crash the whole job if git isn't available; just return empty.
        return ""
    return p.stdout.strip()


def get_git_commit_sha(repo_dir: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo_dir)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_last_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_last_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def make_unique_version_label(prefix: str, commit_sha: str) -> str:
    # Unique + readable: auto-YYYYMMDD-HHMMSS-<sha7> (UTC)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha7 = (commit_sha or "nosha")[:7]
    return f"{prefix}{ts}-{sha7}"


# ----------------------------
# API call
# ----------------------------
def call_versions_create(
    base_url: str,
    api_key: Optional[str],
    data_commons_key: str,
    payload: Dict[str, Any],
    timeout: int = 30,
    query: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{base_url}/data-models/{data_commons_key}/versions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    t0 = time.time()
    #resp = requests.post(url, params=(query or {}), json=payload, headers=headers, timeout=timeout)
    body = json.dumps(payload)
    resp = requests.post(url, data=body, headers=headers, timeout=timeout)
    elapsed = round(time.time() - t0, 3)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP {resp.status_code}: {e}\nResponse text:\n{resp.text}", file=sys.stderr)
        raise

    data = resp.json()
    # Unwrap Lambda proxy if present
    if isinstance(data, dict) and "body" in data:
        body_json = data["body"]
        data = json.loads(body_json) if isinstance(body_json, str) else body_json

    data["_latency_s"] = elapsed
    return data


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", default=".", help="Path to local ccdi-model repo checkout")
    ap.add_argument("--yaml-path", default="ccdi-model-props.yml", help="Path (relative to repo) to YAML file")
    ap.add_argument("--state-path", default=".cron_state/ccdi_publish_state.json", help="Where to store last published SHA/file hash")
    ap.add_argument("--base-url", default=BASE_URL_DEFAULT)
    ap.add_argument("--data-commons-key", default=DATA_COMMONS_KEY_DEFAULT)
    ap.add_argument("--api-key", default=os.getenv("API_KEY"), help="API key (or set API_KEY env var)")
    ap.add_argument("--notes", default="Automated publish from cron job", help="Notes stored with version")
    ap.add_argument("--prefix", default="auto-", help="Version label prefix")
    ap.add_argument("--reference-version", default=None, help="Optional reference version label (e.g., v16)")
    ap.add_argument("--is-default", action="store_true", help="Set the new version as default")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="Publish even if no detected change")
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    yaml_path = (repo_dir / args.yaml_path).resolve()
    state_path = (repo_dir / args.state_path).resolve()

    if not yaml_path.exists():
        raise SystemExit(f"YAML not found: {yaml_path}")

    commit_sha = get_git_commit_sha(repo_dir)  # read-only
    file_hash = sha256_file(yaml_path)

    last = load_last_state(state_path)
    last_sha = last.get("commit_sha")
    last_hash = last.get("yaml_sha256")

    changed = (commit_sha and commit_sha != last_sha) or (file_hash != last_hash)

    if not changed and not args.force:
        print("No change detected (same commit + YAML hash). Skipping publish.")
        return

    model = parse_ccdi_yaml_to_bdi_model(yaml_path)
    version_label = make_unique_version_label(args.prefix, commit_sha)

    payload: Dict[str, Any] = {
        "version_label": version_label,
        "notes": args.notes,
        "is_default": bool(args.is_default),
        "reference_version": args.reference_version,
        "model": model,
    }

    # Omit reference_version if None (often cleaner for APIs)
    if payload["reference_version"] is None:
        payload.pop("reference_version", None)

    result = call_versions_create(
        base_url=args.base_url,
        api_key=args.api_key,
        data_commons_key=args.data_commons_key,
        payload=payload,
        timeout=args.timeout,
    )

    print("Published version:")
    print(json.dumps(result, indent=2))

    save_last_state(
        state_path,
        {
            "commit_sha": commit_sha,
            "yaml_sha256": file_hash,
            "published_version_label": version_label,
        },
    )
    print(f"State saved to: {state_path}")


if __name__ == "__main__":
    main()

