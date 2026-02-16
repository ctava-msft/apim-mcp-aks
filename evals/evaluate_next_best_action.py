#!/usr/bin/env python3
"""
Next Best Action (NBA) Agent Evaluation Script

This script evaluates an agentic "Next Best Action" agent using the Azure AI Evaluation SDK
agent evaluators deployed on the AKS cluster via MCP tools:
- IntentResolutionEvaluator: Measures if agent correctly identifies user intent (score 1-5)
- ToolCallAccuracyEvaluator: Measures if agent made correct tool calls (score 1-5)
- TaskAdherenceEvaluator: Measures if agent response adheres to its assigned tasks (flagged true/false)

The evaluations run ON THE AKS CLUSTER which has access to Azure OpenAI via private endpoint.
This script calls the evaluation MCP tools via APIM + MCP + AKS.

INSTALLATION:
    pip install aiohttp python-dotenv

USAGE EXAMPLES:
    # Dataset mode - evaluate from JSONL file:
    python evaluate_next_best_action_agent.py --data nba_eval_data.jsonl --out ./results

    # Single evaluation mode - evaluate one query/response:
    python evaluate_next_best_action_agent.py --query "Analyze customer churn" --response "Here is my plan..." --out ./results

    # Target mode - run agent live for each query then evaluate:
    python evaluate_next_best_action_agent.py --target --out ./results

    # With custom thresholds:
    python evaluate_next_best_action_agent.py --data nba_eval_data.jsonl --out ./results \
        --intent-threshold 4 --tool-threshold 3 --task-threshold 4

    # Strict mode (exit 1 if thresholds not met):
    python evaluate_next_best_action_agent.py --data nba_eval_data.jsonl --out ./results --strict

    # Direct mode (port-forward, no APIM):
    python evaluate_next_best_action_agent.py --data nba_eval_data.jsonl --out ./results --direct

SAMPLE JSONL ROW (for --data mode):
{
    "query": "What's the next best action for account #12345?",
    "response": "Based on the account analysis, I recommend...",
    "tool_calls": [
        {"type": "tool_call", "tool_call_id": "call_001", "name": "next_best_action", "arguments": {"task": "..."}}
    ],
    "system_message": "You are a Next Best Action agent..."
}

Author: Azure AI Evaluation Script
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default thresholds (1-5 scale for scores, higher is better)
DEFAULT_THRESHOLDS = {
    "intent_resolution": 3,
    "tool_call_accuracy": 3,
    "task_adherence": 3,  # Note: task_adherence uses flagged (true=fail, false=pass)
}

# Configuration file path
CONFIG_FILE = Path(__file__).parent.parent / 'tests' / 'mcp_test_config.json'


# =============================================================================
# MCP CLIENT (adapted from test_next_best_action.py)
# =============================================================================

class MCPClient:
    """MCP Client that maintains SSE session for communicating with the agent."""
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session: Optional[aiohttp.ClientSession] = None
        self.sse_response = None
        self.session_message_url: Optional[str] = None
    
    async def __aenter__(self):
        cookie_jar = aiohttp.CookieJar()
        headers = {}
        if self.auth_token and not self.auth_token.startswith('direct-mode'):
            headers['Authorization'] = f'Bearer {self.auth_token}'
        self.session = aiohttp.ClientSession(
            cookie_jar=cookie_jar,
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sse_response and not self.sse_response.closed:
            self.sse_response.close()
        if self.session:
            await self.session.close()
    
    async def establish_sse_session(self) -> bool:
        """Establish SSE connection and extract session URL."""
        try:
            logger.info(f"Establishing SSE session to: {self.base_url}/sse")
            
            self.sse_response = await self.session.get(
                f'{self.base_url}/sse',
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
            logger.debug(f"SSE Response Status: {self.sse_response.status}")
            
            if self.sse_response.status == 200:
                async for chunk in self.sse_response.content.iter_chunked(1024):
                    if chunk:
                        data = chunk.decode('utf-8', errors='ignore')
                        match = re.search(r'data: (message\?[^\n\r]+)', data)
                        if match:
                            session_path = match.group(1)
                            self.session_message_url = f"{self.base_url}/{session_path}"
                            logger.info(f"Got session URL: {self.session_message_url}")
                            return True
                        break
                
                logger.warning("SSE connected but no session URL found")
                return False
            else:
                response_text = await self.sse_response.text()
                logger.error(f"SSE connection failed: {response_text}")
                return False
                
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
            return False
    
    async def send_request(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 300  # 5 minute timeout for evaluations
    ) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request."""
        request_id = f"eval-request-{datetime.now().timestamp()}"
        
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params:
            jsonrpc_request["params"] = params
        
        message_url = self.session_message_url if self.session_message_url else f'{self.base_url}/message'
        
        try:
            async with self.session.post(
                message_url,
                json=jsonrpc_request,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response", "raw": response_text}
                else:
                    return {"error": f"HTTP {response.status}", "body": response_text}
                    
        except asyncio.TimeoutError:
            return {"error": "Request timed out (evaluations may take a while)"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        """List available MCP tools."""
        result = await self.send_request("tools/list")
        if 'error' not in result:
            return result.get('result', {}).get('tools', [])
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        return await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })


async def get_mcp_token(session: aiohttp.ClientSession, token_url: str) -> Optional[str]:
    """Get MCP access token from APIM OAuth endpoint."""
    logger.info(f"Getting MCP access token from: {token_url}")
    
    try:
        async with session.post(token_url, data={}) as response:
            if response.status == 200:
                data = await response.json()
                token = data.get('access_token', '')
                if token:
                    logger.info(f"Got MCP access token: {token[:30]}...")
                    return token
            logger.error(f"Token request failed: {response.status}")
            return None
    except Exception as e:
        logger.error(f"Error getting token: {e}")
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return data


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load MCP configuration from JSON file."""
    if config_path is None:
        config_path = CONFIG_FILE
    
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file not found: {config_path}")
        return {}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

async def check_evaluation_status(client: MCPClient) -> Dict[str, Any]:
    """Check if evaluation tools are available on the server."""
    result = await client.call_tool("get_evaluation_status", {})
    
    if 'error' in result:
        return {"available": False, "error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    
    if content:
        try:
            status_text = content[0].get('text', '{}')
            return json.loads(status_text)
        except json.JSONDecodeError:
            return {"available": False, "error": "Invalid response"}
    
    return {"available": False, "error": "No content"}


async def run_single_evaluation(
    client: MCPClient,
    query: str,
    response: str,
    tool_calls: Optional[List[Dict]] = None,
    system_message: Optional[str] = None,
    thresholds: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Run a single evaluation via MCP tool."""
    arguments = {
        "query": query,
        "response": response
    }
    if tool_calls:
        arguments["tool_calls"] = tool_calls
    if system_message:
        arguments["system_message"] = system_message
    if thresholds:
        arguments["thresholds"] = thresholds
    
    result = await client.call_tool("run_agent_evaluation", arguments)
    
    if 'error' in result:
        return {"error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        return {"error": error_text}
    
    if content:
        try:
            return json.loads(content[0].get('text', '{}'))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response"}
    
    return {"error": "No content in response"}


async def run_batch_evaluation(
    client: MCPClient,
    evaluation_data: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Run batch evaluation via MCP tool."""
    arguments = {
        "evaluation_data": evaluation_data
    }
    if thresholds:
        arguments["thresholds"] = thresholds
    
    logger.info(f"Running batch evaluation on {len(evaluation_data)} items...")
    result = await client.call_tool("run_batch_evaluation", arguments)
    
    if 'error' in result:
        return {"error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        return {"error": error_text}
    
    if content:
        try:
            return json.loads(content[0].get('text', '{}'))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response"}
    
    return {"error": "No content in response"}


async def run_sequential_evaluation(
    base_url: str,
    token: str,
    evaluation_data: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Evaluate items one at a time, each with a fresh SSE session.

    This avoids SSE timeout issues that occur when a batch evaluation
    takes longer than the SSE connection lifetime (~2 minutes).
    """
    per_item_results = []
    total = len(evaluation_data)
    intent_scores = []
    tool_scores = []
    task_flags = []

    for i, item in enumerate(evaluation_data, 1):
        print(f"\n{'─' * 50}")
        print(f"[{i}/{total}] Evaluating: {item.get('query', '')[:60]}...")

        # Wait between items to avoid rate limits and let the pod recover
        if i > 1:
            wait_secs = 15
            print(f"   Waiting {wait_secs}s between evaluations (rate-limit cooldown)...")
            await asyncio.sleep(wait_secs)

        # Fresh SSE session for each item
        async with MCPClient(base_url, token) as client:
            if not await client.establish_sse_session():
                per_item_results.append({"error": "SSE session failed", "query": item.get("query", "")})
                continue
            await asyncio.sleep(1)

            result = await run_single_evaluation(
                client=client,
                query=item.get("query", ""),
                response=item.get("response", ""),
                tool_calls=item.get("tool_calls"),
                system_message=item.get("system_message"),
                thresholds=thresholds,
            )

        if "error" in result:
            print(f"   [FAIL] {result['error']}")
            per_item_results.append(result)
            continue

        per_item_results.append(result)

        # Collect scores for summary — results may nest under "evaluations"
        evals = result.get("evaluations", result)
        intent = evals.get("intent_resolution", {}).get("score")
        tool = evals.get("tool_call_accuracy", {}).get("score")
        task = evals.get("task_adherence", {}).get("flagged")

        if intent is not None:
            intent_scores.append(float(intent))
            print(f"   Intent: {intent}/5", end="")
        if tool is not None:
            tool_scores.append(float(tool))
            print(f"  Tool: {tool}/5", end="")
        if task is not None:
            task_flags.append(task)
            print(f"  Task: {'flagged' if task else 'pass'}", end="")
        print()

    # Build aggregated summary matching batch format
    avg_intent = round(sum(intent_scores) / len(intent_scores), 2) if intent_scores else 0
    avg_tool = round(sum(tool_scores) / len(tool_scores), 2) if tool_scores else 0
    flagged_count = sum(1 for f in task_flags if f)

    all_passed = True
    if thresholds:
        if avg_intent < thresholds.get("intent_resolution", 0):
            all_passed = False
        if avg_tool < thresholds.get("tool_call_accuracy", 0):
            all_passed = False
        if flagged_count > 0:
            all_passed = False

    return {
        "mode": "sequential",
        "summary": {
            "total_evaluated": total,
            "avg_intent_resolution": avg_intent,
            "avg_tool_call_accuracy": avg_tool,
            "task_adherence_flagged": flagged_count,
            "all_passed": all_passed,
        },
        "per_item_results": per_item_results,
    }


async def run_agent_and_evaluate(
    client: MCPClient,
    query: str,
    thresholds: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Run the NBA agent for a query, then evaluate the response."""
    # First, call the NBA agent
    logger.info(f"Calling next_best_action for: {query[:50]}...")
    agent_result = await client.call_tool("next_best_action", {"task": query})
    
    if 'error' in agent_result:
        return {"error": f"Agent call failed: {agent_result['error']}"}
    
    tool_result = agent_result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        return {"error": f"Agent error: {error_text}"}
    
    if not content:
        return {"error": "No content from agent"}
    
    # Extract agent response
    try:
        agent_response_text = content[0].get('text', '{}')
        agent_response = json.loads(agent_response_text)
    except json.JSONDecodeError:
        agent_response_text = content[0].get('text', '')
        agent_response = {"raw": agent_response_text}
    
    # Build response string for evaluation
    response_parts = []
    if agent_response.get('intent'):
        response_parts.append(f"Intent: {agent_response['intent']}")
    
    plan = agent_response.get('plan', {})
    steps = plan.get('steps', [])
    if steps:
        response_parts.append(f"\nAction Plan ({len(steps)} steps):")
        for step in steps:
            response_parts.append(f"  Step {step.get('step', '?')}: {step.get('action', '')} - {step.get('description', '')}")
    
    response_text = "\n".join(response_parts) if response_parts else agent_response_text
    
    # Build tool_calls for evaluation
    tool_calls = [{
        "type": "tool_call",
        "tool_call_id": f"call_{agent_response.get('task_id', 'unknown')}",
        "name": "next_best_action",
        "arguments": {"task": query}
    }]
    
    # Now evaluate the response
    logger.info("Evaluating agent response...")
    eval_result = await run_single_evaluation(
        client=client,
        query=query,
        response=response_text,
        tool_calls=tool_calls,
        thresholds=thresholds
    )
    
    # Combine agent response with evaluation
    return {
        "query": query,
        "agent_response": agent_response,
        "evaluation": eval_result
    }


# =============================================================================
# OUTPUT AND REPORTING
# =============================================================================

def save_results(
    results: Dict[str, Any],
    output_dir: Path,
    thresholds: Dict[str, int]
) -> Dict[str, Path]:
    """Save evaluation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results as JSON
    results_file = output_dir / f"eval_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / f"eval_summary_{timestamp}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "thresholds": thresholds,
        "summary": results.get("summary", {}),
        "all_passed": results.get("all_passed", False) if "all_passed" in results else None
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_file}")
    
    return {"results": results_file, "summary": summary_file}


def print_summary_report(
    results: Dict[str, Any],
    thresholds: Dict[str, int],
    strict: bool = False
) -> bool:
    """Print summary report to console. Returns True if all thresholds met."""
    print("\n" + "=" * 70)
    print("[STATS] NBA AGENT EVALUATION SUMMARY")
    print("=" * 70)
    
    summary = results.get("summary", {})
    metrics = summary.get("metrics", {})
    
    print(f"\nTotal Evaluated: {summary.get('total_evaluated', 0)}")
    
    all_passed = True
    
    # Intent Resolution
    if "intent_resolution" in metrics:
        ir = metrics["intent_resolution"]
        threshold = thresholds.get("intent_resolution", 3)
        avg = ir.get("average_score", 0)
        pass_rate = ir.get("pass_rate", 0)
        passed = avg >= threshold
        
        status = "[OK]" if passed else "[FAIL]"
        print(f"\n{status} Intent Resolution:")
        print(f"   Average Score: {avg:.2f} (threshold: {threshold})")
        print(f"   Pass Rate: {pass_rate}%")
        print(f"   Range: {ir.get('min', 0)} - {ir.get('max', 0)}")
        
        if not passed:
            all_passed = False
    
    # Tool Call Accuracy
    if "tool_call_accuracy" in metrics:
        tca = metrics["tool_call_accuracy"]
        threshold = thresholds.get("tool_call_accuracy", 3)
        avg = tca.get("average_score", 0)
        pass_rate = tca.get("pass_rate", 0)
        passed = avg >= threshold
        
        status = "[OK]" if passed else "[FAIL]"
        print(f"\n{status} Tool Call Accuracy:")
        print(f"   Average Score: {avg:.2f} (threshold: {threshold})")
        print(f"   Pass Rate: {pass_rate}%")
        print(f"   Range: {tca.get('min', 0)} - {tca.get('max', 0)}")
        
        if not passed:
            all_passed = False
    
    # Task Adherence
    if "task_adherence" in metrics:
        ta = metrics["task_adherence"]
        pass_rate = ta.get("pass_rate", 0)
        passed_count = ta.get("passed_count", 0)
        failed_count = ta.get("failed_count", 0)
        passed = pass_rate >= 50  # Consider passed if more than half pass
        
        status = "[OK]" if passed else "[FAIL]"
        print(f"\n{status} Task Adherence:")
        print(f"   Pass Rate: {pass_rate}%")
        print(f"   Passed: {passed_count}, Failed: {failed_count}")
        
        if not passed:
            all_passed = False
    
    # For single evaluation results
    if "evaluations" in results:
        evals = results["evaluations"]
        all_passed = results.get("all_passed", True)
        
        for name, eval_data in evals.items():
            if "error" in eval_data:
                print(f"\n[FAIL] {name}: Error - {eval_data['error']}")
                all_passed = False
            elif "skipped" in eval_data:
                print(f"\n[SKIP]  {name}: Skipped - {eval_data.get('reason', '')}")
            else:
                passed = eval_data.get("passed", False)
                status = "[OK]" if passed else "[FAIL]"
                
                if "score" in eval_data:
                    print(f"\n{status} {name}:")
                    print(f"   Score: {eval_data.get('score', 0)} (threshold: {eval_data.get('threshold', 3)})")
                    if eval_data.get('explanation'):
                        print(f"   Explanation: {eval_data['explanation'][:100]}...")
                elif "flagged" in eval_data:
                    print(f"\n{status} {name}:")
                    print(f"   Flagged: {eval_data.get('flagged', False)}")
                    if eval_data.get('reasoning'):
                        print(f"   Reasoning: {eval_data['reasoning'][:100]}...")
                
                if not passed:
                    all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("[OK] All evaluation thresholds met!")
    else:
        print("[FAIL] Some thresholds not met")
        if strict:
            print("   (Strict mode enabled - will exit with code 1)")
    
    print("=" * 70 + "\n")
    
    return all_passed


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NBA Agent using Azure AI Evaluation SDK via MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dataset mode:
  python evaluate_next_best_action_agent.py --data nba_eval_data.jsonl --out ./results

  # Single evaluation:
  python evaluate_next_best_action_agent.py --query "Analyze churn" --response "Plan: ..." --out ./results

  # Target mode (calls agent then evaluates):
  python evaluate_next_best_action_agent.py --target --out ./results

  # Direct mode (port-forward):
  python evaluate_next_best_action_agent.py --data data.jsonl --out ./results --direct
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--data", "-d",
        type=str,
        help="Path to JSONL dataset file (batch mode)"
    )
    mode_group.add_argument(
        "--target",
        action="store_true",
        help="Run agent for test queries then evaluate responses"
    )
    
    # Single evaluation inputs
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to evaluate (for single evaluation)"
    )
    parser.add_argument(
        "--response", "-r",
        type=str,
        help="Response to evaluate (for single evaluation)"
    )
    
    # Output
    parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    # Thresholds
    parser.add_argument(
        "--intent-threshold",
        type=int,
        default=DEFAULT_THRESHOLDS["intent_resolution"],
        help=f"Threshold for intent resolution (default: {DEFAULT_THRESHOLDS['intent_resolution']})"
    )
    parser.add_argument(
        "--tool-threshold",
        type=int,
        default=DEFAULT_THRESHOLDS["tool_call_accuracy"],
        help=f"Threshold for tool call accuracy (default: {DEFAULT_THRESHOLDS['tool_call_accuracy']})"
    )
    parser.add_argument(
        "--task-threshold",
        type=int,
        default=DEFAULT_THRESHOLDS["task_adherence"],
        help=f"Threshold for task adherence (default: {DEFAULT_THRESHOLDS['task_adherence']})"
    )
    
    # MCP configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to MCP config JSON file (default: tests/mcp_test_config.json)"
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use direct mode (port-forward, no APIM auth)"
    )
    
    # Behavior
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any threshold is not met"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Evaluate items one at a time with fresh SSE sessions (avoids SSE timeout on large batches)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    config = load_config(args.config)
    
    # Build thresholds
    thresholds = {
        "intent_resolution": args.intent_threshold,
        "tool_call_accuracy": args.tool_threshold,
        "task_adherence": args.task_threshold,
    }
    
    # Determine connection mode
    if args.direct:
        direct_config = config.get('direct', {})
        base_url = direct_config.get('base_url', 'http://localhost:8000/runtime/webhooks/mcp')
        token = 'direct-mode-no-token-needed'
        print(f"\n[LINK] Direct Mode URL: {base_url}")
        print("   (Using port-forward, no auth required)")
    else:
        apim_config = config.get('apim', {})
        base_url = apim_config.get('base_url', '')
        token_url = apim_config.get('oauth_token_url', '')
        
        if not base_url:
            logger.error("No APIM base URL configured. Check tests/mcp_test_config.json")
            return 1
        
        print(f"\n[LINK] APIM Base URL: {base_url}")
        
        # Get token
        async with aiohttp.ClientSession() as session:
            token = await get_mcp_token(session, token_url)
            if not token:
                logger.error("Failed to get access token")
                return 1
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sequential mode bypasses the long-lived SSE session
    if args.sequential and args.data:
        print(f"\n[FILE] Loading dataset from: {args.data}")
        dataset = load_jsonl_data(args.data)
        print(f"   Loaded {len(dataset)} evaluation items")
        
        eval_data = []
        for row in dataset:
            eval_data.append({
                "query": row.get("query", ""),
                "response": row.get("response", ""),
                "tool_calls": row.get("tool_calls", []),
                "system_message": row.get("system_message", "")
            })
        
        # Build thresholds dict
        thresholds_dict = {
            "intent_resolution": thresholds.get("intent_resolution", 3),
            "tool_call_accuracy": thresholds.get("tool_call_accuracy", 3),
            "task_adherence": thresholds.get("task_adherence", 3),
        }
        
        print(f"\n[NOTE] Sequential mode: evaluating {len(eval_data)} items one at a time")
        results = await run_sequential_evaluation(base_url, token, eval_data, thresholds_dict)
        
        if results and "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            return 1
        
        if results:
            saved_files = save_results(results, output_dir, thresholds)
            all_passed = print_summary_report(results, thresholds, args.strict)
            
            print(f"Results saved to:")
            for name, path in saved_files.items():
                print(f"  {name}: {path}")
            
            if args.strict and not all_passed:
                return 1
        
        return 0
    
    # Connect to MCP server
    async with MCPClient(base_url, token) as client:
        # Establish SSE session
        if not await client.establish_sse_session():
            logger.error("Failed to establish SSE session")
            return 1
        
        print("\n[WAIT] Waiting for session to initialize...")
        await asyncio.sleep(2)
        
        # Check evaluation tools are available
        print("\n[CHECK] Checking evaluation tools availability...")
        status = await check_evaluation_status(client)
        
        if not status.get("evaluation_available"):
            logger.error(f"Evaluation tools not available: {status.get('message', status.get('error', 'Unknown'))}")
            print("\n[WARN]  Make sure the AKS server has azure-ai-evaluation installed")
            print("   and FOUNDRY_PROJECT_ENDPOINT is configured.")
            return 1
        
        print(f"[OK] Evaluation tools available")
        print(f"   Model: {status.get('model_deployment', 'N/A')}")
        print(f"   Evaluators: {', '.join(status.get('evaluators', []))}")
        
        # Run evaluation based on mode
        results = None
        
        if args.data:
            # Batch mode - load JSONL and evaluate
            print(f"\n[FILE] Loading dataset from: {args.data}")
            dataset = load_jsonl_data(args.data)
            print(f"   Loaded {len(dataset)} evaluation items")
            
            # Convert to expected format
            eval_data = []
            for row in dataset:
                eval_data.append({
                    "query": row.get("query", ""),
                    "response": row.get("response", ""),
                    "tool_calls": row.get("tool_calls", []),
                    "system_message": row.get("system_message", "")
                })
            
            print(f"\n[WAIT] Running batch evaluation (this may take several minutes)...")
            results = await run_batch_evaluation(client, eval_data, thresholds)
            
        elif args.query and args.response:
            # Single evaluation mode
            print(f"\n[NOTE] Running single evaluation...")
            results = await run_single_evaluation(
                client=client,
                query=args.query,
                response=args.response,
                thresholds=thresholds
            )
            
        elif args.target:
            # Target mode - run agent for test queries then evaluate
            print("\n[TARGET] Target mode - running agent then evaluating")
            
            test_queries = [
                "Analyze customer churn data and create a predictive model to identify at-risk customers",
                "Set up a CI/CD pipeline for deploying microservices to Kubernetes",
                "Design a REST API for a user management system with authentication",
            ]
            
            all_results = []
            for query in test_queries:
                print(f"\n{'─' * 60}")
                print(f"[NOTE] Query: {query[:50]}...")
                
                result = await run_agent_and_evaluate(client, query, thresholds)
                all_results.append(result)
                
                if "error" in result:
                    print(f"[FAIL] Error: {result['error']}")
                else:
                    eval_result = result.get("evaluation", {})
                    if eval_result.get("all_passed"):
                        print("[OK] All evaluations passed")
                    else:
                        print("[FAIL] Some evaluations failed")
            
            # Aggregate results
            results = {
                "mode": "target",
                "summary": {
                    "total_evaluated": len(all_results)
                },
                "per_query_results": all_results
            }
        else:
            # No mode specified - just check status
            print("\n[WARN]  No evaluation mode specified.")
            print("   Use --data, --target, or --query/--response")
            return 0
        
        # Handle errors
        if results and "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            return 1
        
        # Save results
        if results:
            saved_files = save_results(results, output_dir, thresholds)
            
            # Print summary
            all_passed = print_summary_report(results, thresholds, args.strict)
            
            print(f"Results saved to:")
            for name, path in saved_files.items():
                print(f"  {name}: {path}")
            
            if args.strict and not all_passed:
                return 1
        
        return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n\n[WARN]  Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

