#!/usr/bin/env python3
"""Deploy a fine-tuned model after training completes.

This script:
1. Checks the training run status (must be succeeded)
2. Promotes the fine-tuned model via lightning_promote_deployment MCP tool
3. Updates the AKS deployment with USE_TUNED_MODEL and TUNED_MODEL_NAME env vars
4. Restarts the deployment and waits for rollout

Usage:
    python scripts/deploy_finetuned_model.py --training-run-id <id> [--port 8000]
"""

import argparse
import json
import subprocess
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


def mcp_call(base_url: str, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool and return the parsed result."""
    session_url = get_session_url(base_url)
    resp = requests.post(
        session_url,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
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


def kubectl(*args: str) -> str:
    """Run a kubectl command and return stdout."""
    cmd = ["kubectl"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model")
    parser.add_argument("--training-run-id", required=True, help="Training run ID")
    parser.add_argument("--agent-id", default="mcp-agents", help="Agent ID (default: mcp-agents)")
    parser.add_argument("--port", type=int, default=8000, help="MCP port-forward port (default: 8000)")
    parser.add_argument("--namespace", default="mcp-agents", help="K8s namespace (default: mcp-agents)")
    parser.add_argument("--deployment", default="mcp-agents", help="K8s deployment name (default: mcp-agents)")
    parser.add_argument("--skip-promote", action="store_true", help="Skip MCP promote (already promoted)")
    parser.add_argument("--skip-k8s", action="store_true", help="Skip Kubernetes deployment update")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}/runtime/webhooks/mcp"

    # Step 1: Check training run status
    print("=" * 60)
    print("Step 1: Checking training run status...")
    data = mcp_call(base_url, "lightning_get_training_status", {
        "training_run_id": args.training_run_id,
        "agent_id": args.agent_id,
    })

    status = data.get("status")
    model_name = data.get("tuned_model_name")

    if status != "succeeded":
        print(f"ERROR: Training run status is '{status}', expected 'succeeded'")
        if status == "running":
            print("Use scripts/monitor_training.py to wait for completion.")
        return 1

    if not model_name:
        print("ERROR: No tuned model name found in training run")
        return 1

    print(f"  Status: {status}")
    print(f"  Model:  {model_name}")
    print(f"  Metrics: {json.dumps(data.get('metrics', {}))}")

    # Step 2: Promote the model via MCP
    if not args.skip_promote:
        print("\n" + "=" * 60)
        print("Step 2: Promoting fine-tuned model...")
        try:
            promote_result = mcp_call(base_url, "lightning_promote_deployment", {
                "training_run_id": args.training_run_id,
                "agent_id": args.agent_id,
            })
            print(f"  Deployment ID: {promote_result.get('deployment_id')}")
            print(f"  Is Active: {promote_result.get('is_active')}")
            print(f"  Promoted At: {promote_result.get('promoted_at')}")
        except Exception as e:
            print(f"  WARNING: Promote call failed: {e}")
            print("  Continuing with K8s deployment anyway...")
    else:
        print("\nStep 2: Skipping promote (--skip-promote)")

    # Step 3: Update AKS deployment
    if not args.skip_k8s:
        print("\n" + "=" * 60)
        print(f"Step 3: Updating AKS deployment '{args.deployment}'...")
        try:
            kubectl(
                "set", "env", f"deployment/{args.deployment}",
                "-n", args.namespace,
                f"USE_TUNED_MODEL=true",
                f"TUNED_MODEL_NAME={model_name}",
            )
            print(f"  Set USE_TUNED_MODEL=true")
            print(f"  Set TUNED_MODEL_NAME={model_name}")

            # Wait for rollout
            print("  Waiting for rollout...")
            kubectl(
                "rollout", "status", f"deployment/{args.deployment}",
                "-n", args.namespace,
                "--timeout=120s",
            )
            print("  Rollout complete!")

            # Verify env vars
            env_output = kubectl(
                "get", f"deploy/{args.deployment}",
                "-n", args.namespace,
                "-o", "jsonpath={range .spec.template.spec.containers[0].env[*]}{.name}={.value}{\"\\n\"}{end}",
            )
            tuned_vars = [line for line in env_output.split("\n") if "TUNED" in line or "USE_TUNED" in line]
            print("  Verified env vars:")
            for v in tuned_vars:
                print(f"    {v}")

        except Exception as e:
            print(f"  ERROR: K8s update failed: {e}")
            return 1
    else:
        print("\nStep 3: Skipping K8s update (--skip-k8s)")

    # Summary
    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Fine-tuned model: {model_name}")
    print(f"  Deployment: {args.deployment} in namespace {args.namespace}")
    print(f"\nNext step: Run post-training evaluation:")
    print(f"  ..\\.venv\\Scripts\\Activate.ps1")
    print(f"  python -m evals.evaluate_next_best_action \\")
    print(f"    --data evals/next_best_action_eval_data.jsonl \\")
    print(f"    --out evals/eval_results --direct --strict")

    return 0


if __name__ == "__main__":
    sys.exit(main())
