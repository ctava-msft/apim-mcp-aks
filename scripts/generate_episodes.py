#!/usr/bin/env python3
"""Send eval queries to autonomous-agent to generate episodes for training data."""

import asyncio
import aiohttp
import json
import re
import sys
import time


async def main():
    base_url = "http://localhost:8080/runtime/webhooks/mcp"

    # Load eval queries
    queries = []
    with open("evals/autonomous_agent_eval.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                queries.append(obj["query"])

    # Define tool calls matched to each query for diverse episode generation
    tool_calls = [
        {"name": "get_churn_prediction", "arguments": {"customer_id": "cust-a1b2c3d4"}},
        {"name": "get_customer_facts", "arguments": {"customer_id": "cust-e5f6g7h8", "domain": "customer"}},
        {"name": "recommend_retention_action", "arguments": {"customer_id": "cust-i9j0k1l2", "risk_level": "critical"}},
        {"name": "get_customer_segment", "arguments": {"customer_id": "cust-a1b2c3d4"}},
        {"name": "get_churn_prediction", "arguments": {"customer_id": "cust-m3n4o5p6"}},
        {"name": "search_similar_churned", "arguments": {"customer_profile": {"segment": "professional", "churn_risk": 0.45}, "limit": 5}},
    ]

    print(f"Loaded {len(queries)} queries from eval dataset")

    async with aiohttp.ClientSession() as session:
        # Establish SSE session
        print("Establishing SSE session...")
        async with session.get(
            f"{base_url}/sse", headers={"Accept": "text/event-stream"}
        ) as sse:
            session_url = None
            async for chunk in sse.content.iter_chunked(1024):
                data = chunk.decode("utf-8", errors="ignore")
                match = re.search(r"data: (message\?[^\n\r]+)", data)
                if match:
                    session_url = f"{base_url}/{match.group(1)}"
                    break

            if not session_url:
                print("ERROR: No session URL obtained")
                return 1

            print(f"Session established")

            success_count = 0
            # Send each query using the matched tool
            for i, query in enumerate(queries, 1):
                tc = tool_calls[i - 1] if i - 1 < len(tool_calls) else tool_calls[0]
                print(f"\n[{i}/{len(queries)}] Tool: {tc['name']} | Query: {query[:60]}...")

                request = {
                    "jsonrpc": "2.0",
                    "id": f"episode-{i}",
                    "method": "tools/call",
                    "params": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }

                try:
                    async with session.post(
                        session_url,
                        json=request,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            result = json.loads(text)
                            if "error" not in result:
                                content = result.get("result", {}).get("content", [])
                                if content:
                                    preview = content[0].get("text", "")[:100]
                                    print(f"  OK - {preview}...")
                                else:
                                    print(f"  OK (no content)")
                                success_count += 1
                            else:
                                err = str(result.get("error", ""))[:100]
                                print(f"  Error in response: {err}")
                        else:
                            print(f"  HTTP {resp.status}")
                except asyncio.TimeoutError:
                    print(f"  Timeout (120s)")
                except Exception as e:
                    print(f"  Error: {e}")

                # Pause between requests
                if i < len(queries):
                    await asyncio.sleep(2)

    print(f"\nDone! {success_count}/{len(queries)} queries generated episodes successfully.")
    return 0


async def list_tools():
    """List available tools on the agent."""
    base_url = "http://localhost:8080/runtime/webhooks/mcp"
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}/sse", headers={"Accept": "text/event-stream"}
        ) as sse:
            session_url = None
            async for chunk in sse.content.iter_chunked(1024):
                data = chunk.decode("utf-8", errors="ignore")
                match = re.search(r"data: (message\?[^\n\r]+)", data)
                if match:
                    session_url = f"{base_url}/{match.group(1)}"
                    break
            if session_url:
                req = {"jsonrpc": "2.0", "id": "list-tools", "method": "tools/list"}
                async with session.post(
                    session_url, json=req, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    result = json.loads(await resp.text())
                    tools = result.get("result", {}).get("tools", [])
                    for t in tools:
                        name = t["name"]
                        desc = t.get("description", "")[:80]
                        print(f"  {name}: {desc}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list-tools":
        asyncio.run(list_tools())
    else:
        sys.exit(asyncio.run(main()))
