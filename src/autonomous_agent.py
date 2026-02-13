"""
AKS Customer Churn Analysis Agent (Autonomous)
FastAPI MCP Server
Implements Model Context Protocol (MCP) with SSE support
Based on CCA-001 specification: Customer Churn Analysis Agent

This autonomous agent leverages predictive modeling and real-time signals
to proactively identify churn risk and recommend retention actions.
Integrates with Fabric IQ for ontology-grounded customer facts.
"""

import json
import logging
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AKS Customer Churn Analysis MCP Server",
    description="Model Context Protocol Server for Customer Churn Analysis Agent (CCA-001)",
    version="1.0.0"
)

# Azure AI Foundry configuration
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
FOUNDRY_MODEL_DEPLOYMENT_NAME = os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-5.2-chat")

# In-memory session storage (replace with Redis for production)
sessions: Dict[str, Dict[str, Any]] = {}


# ===========================================================================
#  MCP Data Classes
# ===========================================================================

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPToolResult:
    """MCP Tool execution result"""
    content: list
    isError: bool = False


# ===========================================================================
#  Tool Implementations
# ===========================================================================

def get_customer_facts(customer_id: str, domain: str = "customer") -> Dict[str, Any]:
    """
    Retrieve customer facts from Fabric IQ.

    Args:
        customer_id: The customer identifier
        domain: The ontology domain (default: "customer")

    Returns:
        Customer facts including tenure, segment, engagement metrics
    """
    logger.info(f"Retrieving facts for customer {customer_id} in domain {domain}")
    # Simulated customer facts from Fabric IQ ontology
    return {
        "customer_id": customer_id,
        "domain": domain,
        "facts": {
            "name": f"Customer-{customer_id}",
            "segment": "business",
            "tenure_months": 18,
            "status": "active",
            "monthly_spend": 450.0,
            "support_tickets_last_30d": 3,
            "login_frequency": "declining",
            "last_interaction": "2026-01-28",
            "contract_end_date": "2026-06-15",
            "nps_score": 6,
        },
        "relationships": {
            "has_transactions": 142,
            "has_engagement_events": 87,
        },
        "retrieved_at": datetime.utcnow().isoformat(),
    }


def get_churn_prediction(customer_id: str) -> Dict[str, Any]:
    """
    Get churn risk score and drivers for a customer.

    Args:
        customer_id: The customer identifier

    Returns:
        Churn risk score (0-1) and contributing factors
    """
    logger.info(f"Running churn prediction for customer {customer_id}")
    return {
        "customer_id": customer_id,
        "churn_score": 0.82,
        "risk_level": "critical",
        "drivers": [
            {"factor": "declining_login_frequency", "weight": 0.35},
            {"factor": "low_nps_score", "weight": 0.25},
            {"factor": "high_support_tickets", "weight": 0.20},
            {"factor": "approaching_contract_end", "weight": 0.20},
        ],
        "model_version": "churn-v2.1",
        "predicted_at": datetime.utcnow().isoformat(),
    }


def get_customer_segment(customer_id: str) -> Dict[str, Any]:
    """
    Retrieve customer segmentation information.

    Args:
        customer_id: The customer identifier

    Returns:
        Customer segment details and characteristics
    """
    logger.info(f"Retrieving segment for customer {customer_id}")
    return {
        "customer_id": customer_id,
        "segment": "business",
        "sub_segment": "mid-market",
        "lifetime_value": 15200.0,
        "acquisition_channel": "enterprise_sales",
        "characteristics": {
            "avg_monthly_spend": 450.0,
            "product_usage_score": 0.65,
            "feature_adoption_rate": 0.40,
        },
    }


def search_similar_churned(customer_profile: Dict[str, Any], limit: int = 5) -> Dict[str, Any]:
    """
    Find similar customers who have already churned.

    Args:
        customer_profile: Profile attributes to match against
        limit: Maximum number of similar customers to return

    Returns:
        List of similar churned customers with similarity scores
    """
    logger.info(f"Searching for similar churned customers (limit={limit})")
    return {
        "query_profile": customer_profile,
        "similar_churned": [
            {
                "customer_id": "CHURN-4421",
                "similarity": 0.91,
                "churn_date": "2025-11-15",
                "reason": "competitor_switch",
                "tenure_months": 16,
            },
            {
                "customer_id": "CHURN-3298",
                "similarity": 0.87,
                "churn_date": "2025-10-22",
                "reason": "cost_reduction",
                "tenure_months": 20,
            },
        ][:limit],
        "total_matches": 2,
    }


def recommend_retention_action(customer_id: str, risk_level: str) -> Dict[str, Any]:
    """
    Generate personalised retention recommendation based on risk level.

    Args:
        customer_id: The customer identifier
        risk_level: Risk classification (critical / elevated / normal)

    Returns:
        Recommended retention actions
    """
    logger.info(f"Generating retention recommendation for {customer_id} (risk={risk_level})")

    actions_by_risk = {
        "critical": {
            "priority": "immediate",
            "actions": [
                {"type": "personal_outreach", "description": "Schedule executive-level call within 48h"},
                {"type": "discount_offer", "description": "Offer 20% renewal discount"},
                {"type": "feature_unlock", "description": "Grant access to premium features for 30 days"},
            ],
            "escalation": "Account Manager + VP Sales",
        },
        "elevated": {
            "priority": "this_week",
            "actions": [
                {"type": "check_in_email", "description": "Send personalised check-in email"},
                {"type": "training_session", "description": "Schedule product training session"},
            ],
            "escalation": "Customer Success Manager",
        },
        "normal": {
            "priority": "standard",
            "actions": [
                {"type": "nurture_campaign", "description": "Add to engagement nurture campaign"},
            ],
            "escalation": None,
        },
    }

    recommendation = actions_by_risk.get(risk_level, actions_by_risk["normal"])
    return {
        "customer_id": customer_id,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "generated_at": datetime.utcnow().isoformat(),
    }


def create_retention_case(customer_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a case in the CRM for retention outreach.

    Args:
        customer_id: The customer identifier
        action: The recommended action to take

    Returns:
        CRM case confirmation
    """
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Created retention case {case_id} for customer {customer_id}")
    return {
        "case_id": case_id,
        "customer_id": customer_id,
        "action": action,
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
    }


def log_analysis_result(customer_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Log an analysis result for tracking and audit.

    Args:
        customer_id: The customer identifier
        result: The analysis result to log

    Returns:
        Confirmation of logged result
    """
    log_id = f"LOG-{uuid.uuid4().hex[:8].upper()}"
    logger.info(f"Logged analysis result {log_id} for customer {customer_id}")
    return {
        "log_id": log_id,
        "customer_id": customer_id,
        "result_summary": result,
        "logged_at": datetime.utcnow().isoformat(),
    }


# ===========================================================================
#  MCP Tool Catalog (from CCA-001 spec)
# ===========================================================================

TOOLS = [
    MCPTool(
        name="get_customer_facts",
        description="Retrieve customer facts from Fabric IQ",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
                "domain": {"type": "string", "description": "Ontology domain", "default": "customer"},
            },
            "required": ["customer_id"],
        },
    ),
    MCPTool(
        name="get_churn_prediction",
        description="Get churn risk score and drivers",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
            },
            "required": ["customer_id"],
        },
    ),
    MCPTool(
        name="get_customer_segment",
        description="Retrieve customer segmentation",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
            },
            "required": ["customer_id"],
        },
    ),
    MCPTool(
        name="search_similar_churned",
        description="Find similar customers who churned",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_profile": {"type": "object", "description": "Customer profile attributes to match"},
                "limit": {"type": "integer", "description": "Max similar customers to return", "default": 5},
            },
            "required": ["customer_profile"],
        },
    ),
    MCPTool(
        name="recommend_retention_action",
        description="Generate personalised retention recommendation",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
                "risk_level": {"type": "string", "description": "Risk classification: critical, elevated, normal"},
            },
            "required": ["customer_id", "risk_level"],
        },
    ),
    MCPTool(
        name="create_retention_case",
        description="Create case in CRM for outreach",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
                "action": {"type": "object", "description": "Recommended action details"},
            },
            "required": ["customer_id", "action"],
        },
    ),
    MCPTool(
        name="log_analysis_result",
        description="Log analysis result for tracking",
        inputSchema={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "The customer identifier"},
                "result": {"type": "object", "description": "Analysis result to log"},
            },
            "required": ["customer_id", "result"],
        },
    ),
]


# ===========================================================================
#  Tool Dispatcher
# ===========================================================================

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    """Execute an MCP tool by name."""
    start_time = time.time()
    try:
        result = await _execute_tool_impl(tool_name, arguments)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        result = MCPToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            isError=True,
        )
    duration_ms = int((time.time() - start_time) * 1000)
    logger.info(f"Tool {tool_name} executed in {duration_ms}ms")
    return result


async def _execute_tool_impl(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    """Internal implementation of tool execution."""

    if tool_name == "get_customer_facts":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id is required"}],
                isError=True,
            )
        data = get_customer_facts(customer_id, arguments.get("domain", "customer"))
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "get_churn_prediction":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id is required"}],
                isError=True,
            )
        data = get_churn_prediction(customer_id)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "get_customer_segment":
        customer_id = arguments.get("customer_id")
        if not customer_id:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id is required"}],
                isError=True,
            )
        data = get_customer_segment(customer_id)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "search_similar_churned":
        profile = arguments.get("customer_profile")
        if not profile:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_profile is required"}],
                isError=True,
            )
        limit = arguments.get("limit", 5)
        data = search_similar_churned(profile, limit)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "recommend_retention_action":
        customer_id = arguments.get("customer_id")
        risk_level = arguments.get("risk_level")
        if not customer_id or not risk_level:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id and risk_level are required"}],
                isError=True,
            )
        data = recommend_retention_action(customer_id, risk_level)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "create_retention_case":
        customer_id = arguments.get("customer_id")
        action = arguments.get("action")
        if not customer_id or not action:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id and action are required"}],
                isError=True,
            )
        data = create_retention_case(customer_id, action)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    elif tool_name == "log_analysis_result":
        customer_id = arguments.get("customer_id")
        result = arguments.get("result")
        if not customer_id or result is None:
            return MCPToolResult(
                content=[{"type": "text", "text": "customer_id and result are required"}],
                isError=True,
            )
        data = log_analysis_result(customer_id, result)
        return MCPToolResult(content=[{"type": "text", "text": json.dumps(data, default=str)}])

    else:
        return MCPToolResult(
            content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            isError=True,
        )


# ===========================================================================
#  FastAPI Endpoints
# ===========================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/runtime/webhooks/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """
    SSE endpoint for MCP protocol.
    Establishes a long-lived connection for server-sent events.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New SSE session established: {session_id}")

    sessions[session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "message_queue": asyncio.Queue(),
    }

    async def event_generator():
        try:
            # Send initial connection event with message endpoint
            message_url = f"message?sessionId={session_id}"
            yield f"data: {message_url}\n\n"

            # Keep connection alive and send any queued messages
            while True:
                if session_id not in sessions:
                    break

                try:
                    message = await asyncio.wait_for(
                        sessions[session_id]["message_queue"].get(),
                        timeout=30.0,
                    )
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for session {session_id}")
        finally:
            if session_id in sessions:
                del sessions[session_id]
            logger.info(f"SSE session closed: {session_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/runtime/webhooks/mcp/message")
async def mcp_message_endpoint(request: Request):
    """
    Message endpoint for MCP protocol.
    Handles JSON-RPC 2.0 requests.
    """
    try:
        body = await request.json()
        logger.info(f"Received MCP message: {json.dumps(body)[:200]}")

        jsonrpc_version = body.get("jsonrpc")
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id")

        if jsonrpc_version != "2.0":
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32600, "message": "Invalid Request"},
                    "id": request_id,
                },
            )

        # Handle initialize
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "customer-churn-agent",
                        "version": "1.0.0",
                    },
                },
                "id": request_id,
            }
            return JSONResponse(content=response)

        # Handle tools/list
        elif method == "tools/list":
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in TOOLS
            ]
            response = {
                "jsonrpc": "2.0",
                "result": {"tools": tools_list},
                "id": request_id,
            }
            return JSONResponse(content=response)

        # Handle tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await execute_tool(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "result": asdict(result),
                "id": request_id,
            }
            return JSONResponse(content=response)

        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id,
                },
            )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                "id": body.get("id") if "body" in locals() else None,
            },
        )


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "name": "customer-churn-agent",
        "spec_id": "CCA-001",
        "version": "1.0.0",
        "governance_model": "autonomous",
        "description": "Customer Churn Analysis Agent – identifies at-risk customers and recommends retention actions",
        "tools": [t.name for t in TOOLS],
        "endpoints": {
            "health": "/health",
            "sse": "/runtime/webhooks/mcp/sse",
            "message": "/runtime/webhooks/mcp/message",
        },
    }


# ===========================================================================
#  Startup
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
