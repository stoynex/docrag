#!/usr/bin/env python3
"""Simple end-to-end smoke test for DocRAG API.

Usage:
  1) Start API in another terminal:
       uvicorn backend.main:app --host 0.0.0.0 --port 8000
  2) Run:
       python scripts/api_smoke_test.py

Optional:
  BASE_URL=http://127.0.0.1:8000 python scripts/api_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, method=method, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body) if body else {"error": exc.reason}
        return exc.code, parsed


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main() -> int:
    print(f"Running smoke tests against {BASE_URL}")

    status, health = request("GET", "/health")
    assert_true(status == 200, f"/health expected 200, got {status}")
    assert_true(health.get("status") == "ok", "health payload missing status=ok")

    connect_payload = {
        "repository_name": "Smoke Test Repo",
        "documents": [
            {
                "title": "Security Notes",
                "content": (
                    "All staff must use MFA. Password resets require manager approval. "
                    "Audit logs are retained for 90 days."
                ),
            }
        ],
    }

    status, connected = request("POST", "/repositories/connect", connect_payload)
    assert_true(status == 200, f"/repositories/connect expected 200, got {status}")
    repo_id = connected.get("repository_id")
    assert_true(bool(repo_id), "repository_id missing in connect response")

    status, repos = request("GET", "/repositories")
    assert_true(status == 200, f"/repositories expected 200, got {status}")
    assert_true(any(r.get("repository_id") == repo_id for r in repos.get("repositories", [])), "new repo not listed")

    status, search = request(
        "POST",
        "/search",
        {"repository_id": repo_id, "query": "mfa audit", "top_k": 3},
    )
    assert_true(status == 200, f"/search expected 200, got {status}")
    assert_true(len(search.get("results", [])) > 0, "search returned no results")

    status, summary = request("POST", "/summarize", {"repository_id": repo_id, "query": "security"})
    assert_true(status == 200, f"/summarize expected 200, got {status}")
    assert_true(bool(summary.get("summary")), "summary text is empty")

    status, chat = request(
        "POST",
        "/chat",
        {"repository_id": repo_id, "question": "What authentication controls are required?"},
    )
    assert_true(status == 200, f"/chat expected 200, got {status}")
    assert_true(bool(chat.get("answer")), "chat answer is empty")

    print("✅ Smoke test passed: health, connect, list, search, summarize, and chat endpoints")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
