#!/usr/bin/env python3
"""Monitor a fine-tuning training run until completion.

Polls the lightning_get_training_status MCP tool at regular intervals,
which syncs status from the Azure OpenAI API to Cosmos DB.

Usage:
    python scripts/monitor_training.py --training-run-id <id> [--port 8000] [--interval 30]
"""

import argparse
import json
import re
import sys
import time

import requests


def get_session_url(base_url: str) -> str:
    """Establish SSE session and return the message URL."""
    resp = requests.get(f"{base_url}/sse", stream=True, timeout=15)
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            msg_path = line[6:].strip()
            return f"{base_url}/{msg_path}"
    raise RuntimeError("Failed to obtain SSE session URL")


def check_status(base_url: str, training_run_id: str, agent_id: str = "mcp-agents") -> dict:
    """Call lightning_get_training_status via MCP and return the parsed result."""
    session_url = get_session_url(base_url)
    resp = requests.post(
        session_url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "lightning_get_training_status",
                "arguments": {
                    "training_run_id": training_run_id,
                    "agent_id": agent_id,
                },
            },
        },
        timeout=120,
    )
    result = resp.json()
    content = result.get("result", {}).get("content", [])
    if not content:
        raise RuntimeError(f"Empty response: {result}")
    if result.get("result", {}).get("isError"):
        raise RuntimeError(content[0].get("text", "Unknown error"))
    return json.loads(content[0]["text"])


def main():
    parser = argparse.ArgumentParser(description="Monitor fine-tuning training run")
    parser.add_argument("--training-run-id", required=True, help="Training run ID from Cosmos")
    parser.add_argument("--agent-id", default="mcp-agents", help="Agent ID (default: mcp-agents)")
    parser.add_argument("--port", type=int, default=8000, help="MCP port-forward port (default: 8000)")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default: 30)")
    parser.add_argument("--timeout", type=int, default=120, help="Max wait in minutes (default: 120)")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/runtime/webhooks/mcp"
    max_seconds = args.timeout * 60
    elapsed = 0

    print(f"Monitoring training run: {args.training_run_id}")
    print(f"MCP endpoint: {base_url}")
    print(f"Poll interval: {args.interval}s | Timeout: {args.timeout}m")
    print("-" * 60)

    terminal_states = {"succeeded", "failed", "cancelled"}

    while elapsed < max_seconds:
        try:
            data = check_status(base_url, args.training_run_id, args.agent_id)
        except Exception as e:
            print(f"\n[{time.strftime('%H:%M:%S')}] Error polling status: {e}")
            time.sleep(args.interval)
            elapsed += args.interval
            continue

        status = data.get("status", "unknown")
        model = data.get("tuned_model_name")
        metrics = data.get("metrics", {})
        error = data.get("error_message")
        aoai_job = data.get("aoai_job_id", "")

        # Format metrics nicely
        metrics_str = ""
        if metrics:
            parts = [f"{k}={v}" for k, v in metrics.items()]
            metrics_str = " | ".join(parts)

        print(f"[{time.strftime('%H:%M:%S')}] Status: {status}"
              f"{f' | AOAI: {aoai_job}' if aoai_job else ''}"
              f"{f' | Metrics: {metrics_str}' if metrics_str else ''}")

        if status in terminal_states:
            print("-" * 60)
            if status == "succeeded":
                print(f"Training SUCCEEDED!")
                print(f"Fine-tuned model: {model}")
                if metrics:
                    print(f"Final metrics: {json.dumps(metrics, indent=2)}")
                print(f"\nNext step: Deploy the model with:")
                print(f"  python scripts/deploy_finetuned_model.py \\")
                print(f"    --training-run-id {args.training_run_id} \\")
                print(f"    --port {args.port}")
                return 0
            elif status == "failed":
                print(f"Training FAILED: {error}")
                return 1
            else:
                print(f"Training CANCELLED")
                return 1

        time.sleep(args.interval)
        elapsed += args.interval

    print(f"\nTimeout reached ({args.timeout} minutes). Training still running.")
    print("Re-run this script to continue monitoring.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
