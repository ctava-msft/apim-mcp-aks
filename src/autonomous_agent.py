"""
AKS Utilization Management Agent
FastAPI MCP Server
Implements Model Context Protocol (MCP) with SSE support
Specialized for Utilization Management (UM) domain: coverage policies,
step therapy, prior authorization, turnaround times, continuity-of-care.
Features AI Search long-term memory for policy knowledge base retrieval.
Includes Agent Lightning for fine-tuning and behavior optimization.
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
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
import os

# Memory Provider imports
from memory import (
    ShortTermMemory, MemoryEntry, MemoryType, CompositeMemory, LongTermMemory,
    AISEARCH_CONTEXT_PROVIDER_AVAILABLE,
)

# Agent Lightning imports (for fine-tuning and behavior optimization)
try:
    from lightning import (
        EpisodeCaptureHook, get_capture_hook,
        DeploymentRegistry, get_deployment_registry,
        RewardWriter, get_reward_writer,
        DatasetBuilder, get_dataset_builder,
        TrainingRunner, get_training_runner,
        RLLedgerCosmos, get_rl_ledger,
        Episode, Reward, RewardSource, Dataset, TrainingRun, TrainingStatus, Deployment,
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AKS Utilization Management MCP Server",
    description="Model Context Protocol Server for Utilization Management Agent with Agentic Retrieval",
    version="1.0.0",
)

# CosmosDB configuration
COSMOSDB_ENDPOINT = os.getenv("COSMOSDB_ENDPOINT", "")
COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME", "mcpdb")
COSMOSDB_SESSIONS_CONTAINER = "um_sessions"

# Initialize CosmosDB client
cosmos_client = None
cosmos_database = None
cosmos_sessions_container = None

if COSMOSDB_ENDPOINT:
    try:
        credential = DefaultAzureCredential()
        cosmos_client = CosmosClient(COSMOSDB_ENDPOINT, credential=credential)
        cosmos_database = cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME)
        cosmos_sessions_container = cosmos_database.get_container_client(COSMOSDB_SESSIONS_CONTAINER)
        logger.info("CosmosDB client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CosmosDB client: {e}")
else:
    logger.warning("COSMOSDB_ENDPOINT not configured - session storage will not work")

# Initialize Memory Providers
short_term_memory: Optional[ShortTermMemory] = None
composite_memory: Optional[CompositeMemory] = None

if COSMOSDB_ENDPOINT:
    try:
        short_term_memory = ShortTermMemory(
            endpoint=COSMOSDB_ENDPOINT,
            database_name=COSMOSDB_DATABASE_NAME,
            container_name="um_short_term_memory",
            default_ttl=3600,
        )
        composite_memory = CompositeMemory(
            short_term=short_term_memory,
            long_term=None,
        )
        logger.info("Memory providers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory providers: {e}")
else:
    logger.warning("COSMOSDB_ENDPOINT not configured - memory providers will not work")

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Azure AI Foundry configuration
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
FOUNDRY_MODEL_DEPLOYMENT_NAME = os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-large")

# Agent Lightning configuration
LIGHTNING_AGENT_ID = os.getenv("LIGHTNING_AGENT_ID", "um-agent")
ENABLE_LIGHTNING_CAPTURE = os.getenv("ENABLE_LIGHTNING_CAPTURE", "false").lower() == "true"
USE_TUNED_MODEL = os.getenv("USE_TUNED_MODEL", "false").lower() == "true"
TUNED_MODEL_DEPLOYMENT_NAME = os.getenv("TUNED_MODEL_DEPLOYMENT_NAME", "")

# Initialize Agent Lightning components
episode_capture_hook: Optional[EpisodeCaptureHook] = None
deployment_registry: Optional[DeploymentRegistry] = None
reward_writer: Optional[RewardWriter] = None
dataset_builder: Optional[DatasetBuilder] = None
training_runner: Optional[TrainingRunner] = None
rl_ledger: Optional[RLLedgerCosmos] = None

if LIGHTNING_AVAILABLE:
    try:
        episode_capture_hook = get_capture_hook()
        deployment_registry = get_deployment_registry()
        reward_writer = get_reward_writer()
        dataset_builder = get_dataset_builder()
        training_runner = get_training_runner()
        rl_ledger = get_rl_ledger()
        logger.info(f"Agent Lightning initialized (capture={ENABLE_LIGHTNING_CAPTURE}, use_tuned={USE_TUNED_MODEL})")
    except Exception as e:
        logger.warning(f"Failed to initialize Agent Lightning: {e}")
else:
    logger.info("Agent Lightning not available - fine-tuning features disabled")


def get_model_deployment() -> str:
    """Get the model deployment name, preferring tuned model if enabled."""
    if not USE_TUNED_MODEL:
        return FOUNDRY_MODEL_DEPLOYMENT_NAME

    if deployment_registry:
        try:
            tuned_model = deployment_registry.get_active_model(LIGHTNING_AGENT_ID)
            if tuned_model:
                return tuned_model
        except Exception as e:
            logger.warning(f"Failed to get tuned model from registry: {e}")

    if TUNED_MODEL_DEPLOYMENT_NAME:
        return TUNED_MODEL_DEPLOYMENT_NAME

    return FOUNDRY_MODEL_DEPLOYMENT_NAME


# Azure AI Search configuration for UM knowledge base
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "um-policies")
AZURE_SEARCH_KNOWLEDGE_BASE_NAME = os.getenv("AZURE_SEARCH_KNOWLEDGE_BASE_NAME", "um-knowledge-base")

# AI Search Long-Term Memory for UM policy documents
long_term_memory: Optional[LongTermMemory] = None


# =========================================
# Embedding Helper
# =========================================

def get_embedding(text: str) -> List[float]:
    """Generate embeddings using Azure AI Foundry text-embedding-3-large."""
    if not FOUNDRY_PROJECT_ENDPOINT:
        raise ValueError("Foundry endpoint not configured")

    from openai import AzureOpenAI

    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    base_endpoint = (
        FOUNDRY_PROJECT_ENDPOINT.split("/api/projects")[0]
        if "/api/projects" in FOUNDRY_PROJECT_ENDPOINT
        else FOUNDRY_PROJECT_ENDPOINT
    )
    client = AzureOpenAI(
        azure_endpoint=base_endpoint,
        api_key=token.token,
        api_version="2024-02-15-preview",
    )
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_DEPLOYMENT_NAME, input=text
    )
    return response.data[0].embedding


# =========================================
# UM System Prompt
# =========================================

UM_SYSTEM_PROMPT = """You are an autonomous AI agent built using the Azure Agents Control Plane.
You operate as a domain-specific Utilization Management (UM) assistant for a U.S. health plan.
You must:
- Follow instructions provided by the agent framework
- Ground all answers in the attached knowledge base(s)
- Use agentic retrieval patterns (planning, source selection, evidence reconciliation)
- Produce grounded, explainable answers suitable for regulated healthcare environments

You are not a chatbot. You are an agent that reasons over policy, operations, and regulatory context.

Your purpose is to assist internal users (UM nurses, care managers, operations staff) by answering questions related to:
- Coverage and medical necessity policies
- Step therapy and exception logic
- Prior authorization requirements
- Operational turnaround times
- Continuity-of-care considerations

You do not make final coverage determinations.
You summarize, reconcile, and explain applicable rules and guidance.

SAFETY CONSTRAINTS:
- Never invent policy rules not present in the knowledge base
- Never override coverage policy with operational guidance
- Never provide member-specific or PHI-based answers
- Clearly state when human review is required
- Surface conflicts explicitly when policies conflict and cannot be reconciled

REASONING PROCESS:
1. Decompose the question into sub-questions
2. Select the appropriate knowledge sources for each sub-question
3. Retrieve evidence from each source independently
4. Reconcile conflicts (prefer newer effective dates, explain exceptions)
5. Synthesize a single, grounded response with citations"""


# =========================================
# AI Search Agentic Retrieval
# =========================================

async def search_knowledge_base(query: str, source_filter: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the UM knowledge base via AI Search long-term memory.
    Supports filtering by source type: coverage_policy, step_therapy,
    exception_logic, operations, regulatory.
    """
    results = []

    if long_term_memory:
        try:
            context = await long_term_memory.get_context(query)
            if context:
                results.append({
                    "source": "ai_search_context",
                    "content": context,
                    "score": 1.0,
                })
        except Exception as e:
            logger.warning(f"AI Search context retrieval failed: {e}")

        try:
            instructions = await long_term_memory.search_task_instructions(
                task_description=query,
                limit=limit,
                include_steps=True,
            )
            for inst in instructions:
                results.append({
                    "source": inst.get("category", "policy"),
                    "title": inst.get("title", ""),
                    "content": inst.get("description", ""),
                    "content_excerpt": inst.get("content_excerpt", ""),
                    "score": inst.get("score", 0),
                    "effective_date": inst.get("effective_date", ""),
                    "steps": inst.get("steps", []),
                })
        except Exception as e:
            logger.warning(f"AI Search instruction retrieval failed: {e}")

    if not results:
        results.append({
            "source": "fallback",
            "content": "No matching policies found in the knowledge base for this query.",
            "score": 0,
        })

    return results


async def decompose_and_retrieve(query: str, member_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Agentic retrieval: decompose a UM question into sub-questions,
    route each to the appropriate knowledge source, retrieve evidence,
    and reconcile results.
    """
    sub_questions = _decompose_question(query, member_context)
    evidence = {}

    for sq in sub_questions:
        source_type = sq.get("source_type", "coverage_policy")
        sq_query = sq.get("question", query)
        results = await search_knowledge_base(sq_query, source_filter=source_type)
        evidence[sq_query] = {
            "source_type": source_type,
            "results": results,
            "result_count": len(results),
        }

    conflicts = _detect_conflicts(evidence)

    return {
        "sub_questions": sub_questions,
        "evidence": evidence,
        "conflicts": conflicts,
        "total_sources": sum(e["result_count"] for e in evidence.values()),
    }


def _decompose_question(query: str, member_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Decompose a UM question into sub-questions with source routing.
    Uses keyword matching for deterministic decomposition.
    """
    sub_questions = []
    query_lower = query.lower()

    if any(kw in query_lower for kw in ["prior auth", "pa required", "authorization"]):
        sub_questions.append({
            "question": f"Prior authorization requirements: {query}",
            "source_type": "coverage_policy",
        })

    if any(kw in query_lower for kw in ["step therapy", "step-therapy", "fail first"]):
        sub_questions.append({
            "question": f"Step therapy requirements and exceptions: {query}",
            "source_type": "step_therapy",
        })

    if any(kw in query_lower for kw in ["exception", "override", "waiver"]):
        sub_questions.append({
            "question": f"Exception conditions and overrides: {query}",
            "source_type": "exception_logic",
        })

    if any(kw in query_lower for kw in ["turnaround", "timeline", "urgent", "expedited"]):
        sub_questions.append({
            "question": f"Operational turnaround times: {query}",
            "source_type": "operations",
        })

    if any(kw in query_lower for kw in ["continuity", "out-of-network", "oon", "transition"]):
        sub_questions.append({
            "question": f"Continuity-of-care considerations: {query}",
            "source_type": "regulatory",
        })

    if any(kw in query_lower for kw in ["medicare", "cms", "regulation", "federal"]):
        sub_questions.append({
            "question": f"Medicare/CMS regulatory context: {query}",
            "source_type": "regulatory",
        })

    if any(kw in query_lower for kw in ["medical necessity", "coverage", "criteria"]):
        sub_questions.append({
            "question": f"Medical necessity criteria: {query}",
            "source_type": "coverage_policy",
        })

    if not sub_questions:
        sub_questions.append({
            "question": query,
            "source_type": "coverage_policy",
        })

    return sub_questions


def _detect_conflicts(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect potential conflicts between evidence from different sources."""
    conflicts = []
    source_types = set()
    for sq, ev in evidence.items():
        source_types.add(ev["source_type"])

    if "coverage_policy" in source_types and "exception_logic" in source_types:
        conflicts.append({
            "type": "policy_exception_overlap",
            "description": "Coverage policy and exception logic may provide differing guidance. "
                          "Verify whether the exception modifies or overrides the base policy.",
            "resolution": "Prefer the exception policy when conditions are met; otherwise follow base policy.",
        })

    if "operations" in source_types and "coverage_policy" in source_types:
        conflicts.append({
            "type": "ops_vs_policy",
            "description": "Operational guidance and coverage policy address different aspects. "
                          "Operational rules (turnaround times) do not override clinical coverage decisions.",
            "resolution": "Apply coverage policy for clinical decisions; use operational guidance for timelines only.",
        })

    return conflicts


def _synthesize_response(query: str, retrieval: Dict[str, Any]) -> str:
    """
    Synthesize a grounded response from retrieval evidence.
    If Foundry is configured, use LLM synthesis; otherwise build
    a structured text summary.
    """
    if FOUNDRY_PROJECT_ENDPOINT:
        try:
            return _llm_synthesize(query, retrieval)
        except Exception as e:
            logger.warning(f"LLM synthesis failed, falling back to structured summary: {e}")

    return _structured_summary(query, retrieval)


def _structured_summary(query: str, retrieval: Dict[str, Any]) -> str:
    """Build a deterministic structured summary from evidence."""
    lines = [f"## UM Agent Response\n\n**Query:** {query}\n"]

    evidence = retrieval.get("evidence", {})
    for sq, ev in evidence.items():
        lines.append(f"\n### {ev['source_type'].replace('_', ' ').title()}")
        lines.append(f"**Sub-question:** {sq}")
        for r in ev.get("results", []):
            if r.get("title"):
                lines.append(f"- **{r['title']}** (score: {r.get('score', 0):.2f})")
            content = r.get("content", r.get("content_excerpt", ""))
            if content:
                lines.append(f"  {content[:300]}")

    conflicts = retrieval.get("conflicts", [])
    if conflicts:
        lines.append("\n### Conflicts Detected")
        for c in conflicts:
            lines.append(f"- **{c['type']}**: {c['description']}")
            lines.append(f"  Resolution: {c['resolution']}")

    lines.append("\n---\n*This response is grounded in the UM knowledge base. "
                 "It does not constitute a coverage determination. Human review may be required.*")

    return "\n".join(lines)


def _llm_synthesize(query: str, retrieval: Dict[str, Any]) -> str:
    """Use LLM to synthesize a grounded answer from retrieval evidence."""
    from openai import AzureOpenAI

    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    base_endpoint = (
        FOUNDRY_PROJECT_ENDPOINT.split("/api/projects")[0]
        if "/api/projects" in FOUNDRY_PROJECT_ENDPOINT
        else FOUNDRY_PROJECT_ENDPOINT
    )
    client = AzureOpenAI(
        azure_endpoint=base_endpoint,
        api_key=token.token,
        api_version="2024-02-15-preview",
    )
    model_deployment = get_model_deployment()

    context_parts = []
    evidence = retrieval.get("evidence", {})
    for sq, ev in evidence.items():
        context_parts.append(f"Source type: {ev['source_type']}")
        for r in ev.get("results", []):
            if r.get("title"):
                context_parts.append(f"Title: {r['title']}")
            content = r.get("content", r.get("content_excerpt", ""))
            if content:
                context_parts.append(content[:500])

    conflicts = retrieval.get("conflicts", [])
    if conflicts:
        context_parts.append("CONFLICTS DETECTED:")
        for c in conflicts:
            context_parts.append(f"- {c['type']}: {c['description']}")

    context_text = "\n".join(context_parts)

    response = client.chat.completions.create(
        model=model_deployment,
        messages=[
            {"role": "system", "content": UM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Based on the following knowledge base evidence, answer this question:\n\n"
                           f"Question: {query}\n\nEvidence:\n{context_text}",
            },
        ],
    )

    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content
    return "No response generated"


# =========================================
# Initialize AI Search Long-Term Memory
# =========================================

def _initialize_long_term_memory():
    """Initialize AI Search long-term memory for UM policy documents."""
    global long_term_memory

    if AZURE_SEARCH_ENDPOINT and FOUNDRY_PROJECT_ENDPOINT:
        try:
            long_term_memory = LongTermMemory(
                search_endpoint=AZURE_SEARCH_ENDPOINT,
                foundry_endpoint=FOUNDRY_PROJECT_ENDPOINT,
                index_name=AZURE_SEARCH_INDEX_NAME,
                knowledge_base_name=AZURE_SEARCH_KNOWLEDGE_BASE_NAME,
                mode="agentic",
            )
            long_term_memory.set_embedding_function(get_embedding)

            if composite_memory:
                composite_memory._long_term = long_term_memory

            if AISEARCH_CONTEXT_PROVIDER_AVAILABLE:
                logger.info(f"LongTermMemory initialized with AzureAISearchContextProvider: {AZURE_SEARCH_INDEX_NAME}")
            else:
                logger.warning(f"LongTermMemory initialized WITHOUT AzureAISearchContextProvider: {AZURE_SEARCH_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize long-term memory: {e}")
    else:
        logger.warning("AZURE_SEARCH_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT not configured - long-term memory will not work")


_initialize_long_term_memory()


# =========================================
# MCP Tool & Result Dataclasses
# =========================================

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


# =========================================
# Domain-Specific UM Tools
# =========================================

TOOLS = [
    # ── UM Domain Tools ──
    MCPTool(
        name="search_coverage_policy",
        description="Search medical necessity and coverage policy documents in the UM knowledge base. Returns grounded policy evidence.",
        inputSchema={
            "type": "object",
            "properties": {
                "condition": {"type": "string", "description": "Patient condition or diagnosis"},
                "service": {"type": "string", "description": "Requested service or procedure"},
                "plan_type": {"type": "string", "description": "Plan type (e.g., Medicare Advantage, Commercial)"},
            },
            "required": ["service"],
        },
    ),
    MCPTool(
        name="search_step_therapy",
        description="Retrieve step therapy requirements and exception logic for a drug or service.",
        inputSchema={
            "type": "object",
            "properties": {
                "drug_or_service": {"type": "string", "description": "Drug or service name"},
                "condition": {"type": "string", "description": "Patient condition"},
            },
            "required": ["drug_or_service"],
        },
    ),
    MCPTool(
        name="check_pa_required",
        description="Determine if prior authorization is required for a service under a given plan.",
        inputSchema={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "Service or procedure name"},
                "plan_type": {"type": "string", "description": "Plan type"},
                "urgency": {"type": "string", "description": "Urgency level: standard or urgent", "enum": ["standard", "urgent"]},
            },
            "required": ["service"],
        },
    ),
    MCPTool(
        name="get_turnaround_time",
        description="Retrieve operational turnaround time for prior authorization decisions.",
        inputSchema={
            "type": "object",
            "properties": {
                "request_type": {"type": "string", "description": "Request type (e.g., prior_auth, appeal)"},
                "urgency": {"type": "string", "description": "Urgency level: standard or urgent", "enum": ["standard", "urgent"]},
            },
            "required": ["request_type"],
        },
    ),
    MCPTool(
        name="search_continuity_of_care",
        description="Retrieve continuity-of-care rules and considerations for provider transitions.",
        inputSchema={
            "type": "object",
            "properties": {
                "provider_network_status": {"type": "string", "description": "Provider network status (in-network, out-of-network)"},
                "transition_type": {"type": "string", "description": "Transition type (e.g., new_enrollment, provider_exit)"},
            },
            "required": ["provider_network_status"],
        },
    ),
    MCPTool(
        name="search_regulatory_guidance",
        description="Search CMS and Medicare Advantage regulatory context.",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Regulatory topic"},
                "regulation_type": {"type": "string", "description": "Type: cms, medicare_advantage, state"},
            },
            "required": ["topic"],
        },
    ),
    MCPTool(
        name="um_answer_question",
        description="Answer a utilization management question using agentic retrieval across clinical policies, operational guidance, and regulatory context. Decomposes the question, retrieves evidence from multiple sources, reconciles conflicts, and synthesizes a grounded response.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The UM question to answer"},
                "member_context": {
                    "type": "object",
                    "description": "Optional member context",
                    "properties": {
                        "plan_type": {"type": "string"},
                        "line_of_business": {"type": "string"},
                        "state": {"type": "string"},
                        "condition": {"type": "string"},
                        "requested_service": {"type": "string"},
                        "provider_network_status": {"type": "string"},
                    },
                },
                "urgency": {"type": "string", "description": "Urgency level", "enum": ["standard", "urgent"]},
            },
            "required": ["query"],
        },
    ),
    # ── Memory Tools ──
    MCPTool(
        name="store_memory",
        description="Store information in short-term memory for later retrieval within a session.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to store"},
                "session_id": {"type": "string", "description": "Session ID"},
                "memory_type": {"type": "string", "description": "Type: context, conversation, task, plan", "enum": ["context", "conversation", "task", "plan"]},
            },
            "required": ["content", "session_id"],
        },
    ),
    MCPTool(
        name="recall_memory",
        description="Recall relevant memories from short-term memory based on semantic similarity.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to search memories"},
                "session_id": {"type": "string", "description": "Session ID"},
                "limit": {"type": "integer", "description": "Max results (default: 5)"},
            },
            "required": ["query", "session_id"],
        },
    ),
    # ── Agent Lightning Tools (RLHF Fine-Tuning) ──
    MCPTool(
        name="lightning_list_episodes",
        description="List captured episodes from Agent Lightning. Episodes represent agent interactions that can be used for RLHF fine-tuning.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Filter by agent ID (default: um-agent)"},
                "limit": {"type": "integer", "description": "Max episodes to return (default: 20)"},
                "start_date": {"type": "string", "description": "Filter after this date (ISO format)"},
                "end_date": {"type": "string", "description": "Filter before this date (ISO format)"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_get_episode",
        description="Get detailed information about a specific episode including tool calls.",
        inputSchema={
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode ID"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
            },
            "required": ["episode_id"],
        },
    ),
    MCPTool(
        name="lightning_assign_reward",
        description="Assign a reward/label to an episode for RLHF training.",
        inputSchema={
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Episode ID"},
                "reward_value": {"type": "number", "description": "Reward value -1.0 to 1.0"},
                "reward_source": {"type": "string", "description": "Source of reward", "enum": ["human_approval", "eval_score", "test_result", "safety_check"]},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "rubric": {"type": "string", "description": "Evaluation rubric"},
                "evaluator": {"type": "string", "description": "Evaluator name"},
                "comments": {"type": "string", "description": "Comments"},
            },
            "required": ["episode_id", "reward_value"],
        },
    ),
    MCPTool(
        name="lightning_list_rewards",
        description="List rewards assigned to episodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "episode_id": {"type": "string", "description": "Filter by episode ID"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "limit": {"type": "integer", "description": "Max rewards to return"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_build_dataset",
        description="Build a fine-tuning dataset from rewarded episodes.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Dataset name"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "description": {"type": "string", "description": "Description"},
                "min_reward": {"type": "number", "description": "Min reward threshold (default: 0.5)"},
            },
            "required": ["name"],
        },
    ),
    MCPTool(
        name="lightning_list_datasets",
        description="List available fine-tuning datasets.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "limit": {"type": "integer", "description": "Max datasets to return"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_start_training",
        description="Start a fine-tuning training run using Azure OpenAI.",
        inputSchema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Dataset ID"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "base_model": {"type": "string", "description": "Base model (default: gpt-4o-mini-2024-07-18)"},
                "n_epochs": {"type": "integer", "description": "Training epochs (default: 3)"},
            },
            "required": ["dataset_id"],
        },
    ),
    MCPTool(
        name="lightning_get_training_status",
        description="Get the status of a training run.",
        inputSchema={
            "type": "object",
            "properties": {
                "training_run_id": {"type": "string", "description": "Training run ID"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
            },
            "required": ["training_run_id"],
        },
    ),
    MCPTool(
        name="lightning_list_training_runs",
        description="List training runs.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "limit": {"type": "integer", "description": "Max runs to return"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_promote_deployment",
        description="Promote a tuned model to active deployment.",
        inputSchema={
            "type": "object",
            "properties": {
                "training_run_id": {"type": "string", "description": "Training run ID"},
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "promoted_by": {"type": "string", "description": "Who is promoting"},
            },
            "required": ["training_run_id"],
        },
    ),
    MCPTool(
        name="lightning_get_active_deployment",
        description="Get the currently active tuned model deployment.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_list_deployments",
        description="List all deployments (active and historical).",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "limit": {"type": "integer", "description": "Max deployments to return"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_rollback_deployment",
        description="Rollback to a previous deployment.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "target_deployment_id": {"type": "string", "description": "Target deployment ID"},
                "reason": {"type": "string", "description": "Rollback reason"},
                "rolled_back_by": {"type": "string", "description": "Who is rolling back"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_deactivate_deployment",
        description="Deactivate the current tuned model deployment, reverting to base model.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
                "reason": {"type": "string", "description": "Deactivation reason"},
            },
            "required": [],
        },
    ),
    MCPTool(
        name="lightning_get_stats",
        description="Get comprehensive Agent Lightning statistics.",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent ID (default: um-agent)"},
            },
            "required": [],
        },
    ),
]


# =========================================
# Tool Execution
# =========================================

async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    """Execute an MCP tool with optional Agent Lightning episode capture."""
    start_time = time.time()
    result = None
    error_message = None

    try:
        result = await _execute_tool_impl(tool_name, arguments)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        error_message = str(e)
        result = MCPToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            isError=True,
        )

    # Capture episode for Agent Lightning
    if episode_capture_hook and episode_capture_hook.is_enabled():
        try:
            duration_ms = int((time.time() - start_time) * 1000)
            result_text = ""
            if result and result.content:
                for item in result.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        result_text += item.get("text", "")

            user_input = f"Call tool '{tool_name}' with arguments: {json.dumps(arguments, default=str)}"
            episode_capture_hook.capture_from_tool_result(
                tool_name=tool_name,
                arguments=arguments,
                result=result_text,
                user_input=user_input,
                model_deployment=get_model_deployment(),
                duration_ms=duration_ms,
            )
        except Exception as capture_error:
            logger.warning(f"Failed to capture episode: {capture_error}")

    return result


async def _execute_tool_impl(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    """Internal tool dispatch."""
    try:
        # ── UM Domain Tools ──
        if tool_name == "search_coverage_policy":
            service = arguments.get("service", "")
            condition = arguments.get("condition", "")
            plan_type = arguments.get("plan_type", "")
            query = f"Coverage policy for {service}"
            if condition:
                query += f" for {condition}"
            if plan_type:
                query += f" under {plan_type}"

            results = await search_knowledge_base(query, source_filter="coverage_policy")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"results": results}, indent=2)}])

        elif tool_name == "search_step_therapy":
            drug_or_service = arguments.get("drug_or_service", "")
            condition = arguments.get("condition", "")
            query = f"Step therapy requirements for {drug_or_service}"
            if condition:
                query += f" for {condition}"

            results = await search_knowledge_base(query, source_filter="step_therapy")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"results": results}, indent=2)}])

        elif tool_name == "check_pa_required":
            service = arguments.get("service", "")
            plan_type = arguments.get("plan_type", "")
            urgency = arguments.get("urgency", "standard")
            query = f"Is prior authorization required for {service}"
            if plan_type:
                query += f" under {plan_type}"
            query += f" ({urgency} request)"

            results = await search_knowledge_base(query, source_filter="coverage_policy")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"pa_check": {"service": service, "plan_type": plan_type, "urgency": urgency, "results": results}}, indent=2)}])

        elif tool_name == "get_turnaround_time":
            request_type = arguments.get("request_type", "")
            urgency = arguments.get("urgency", "standard")
            query = f"Turnaround time for {request_type} ({urgency})"

            results = await search_knowledge_base(query, source_filter="operations")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"turnaround": {"request_type": request_type, "urgency": urgency, "results": results}}, indent=2)}])

        elif tool_name == "search_continuity_of_care":
            provider_status = arguments.get("provider_network_status", "")
            transition_type = arguments.get("transition_type", "")
            query = f"Continuity of care for {provider_status} provider"
            if transition_type:
                query += f" ({transition_type})"

            results = await search_knowledge_base(query, source_filter="regulatory")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"continuity_of_care": {"provider_network_status": provider_status, "transition_type": transition_type, "results": results}}, indent=2)}])

        elif tool_name == "search_regulatory_guidance":
            topic = arguments.get("topic", "")
            regulation_type = arguments.get("regulation_type", "")
            query = f"Regulatory guidance: {topic}"
            if regulation_type:
                query += f" ({regulation_type})"

            results = await search_knowledge_base(query, source_filter="regulatory")
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"regulatory": {"topic": topic, "regulation_type": regulation_type, "results": results}}, indent=2)}])

        elif tool_name == "um_answer_question":
            query = arguments.get("query", "")
            member_context = arguments.get("member_context", {})
            urgency = arguments.get("urgency", "standard")

            if not query:
                return MCPToolResult(content=[{"type": "text", "text": "No query provided"}], isError=True)

            retrieval = await decompose_and_retrieve(query, member_context)
            answer = _synthesize_response(query, retrieval)

            response = {
                "query": query,
                "member_context": member_context,
                "urgency": urgency,
                "answer": answer,
                "retrieval_metadata": {
                    "sub_questions": retrieval["sub_questions"],
                    "total_sources": retrieval["total_sources"],
                    "conflicts": retrieval["conflicts"],
                },
            }
            return MCPToolResult(content=[{"type": "text", "text": json.dumps(response, indent=2)}])

        # ── Memory Tools ──
        elif tool_name == "store_memory":
            if not short_term_memory:
                return MCPToolResult(content=[{"type": "text", "text": "Memory provider not configured"}], isError=True)

            content = arguments.get("content", "")
            session_id = arguments.get("session_id", "")
            memory_type = arguments.get("memory_type", "context")

            type_map = {
                "context": MemoryType.CONTEXT,
                "conversation": MemoryType.CONVERSATION,
                "task": MemoryType.TASK,
                "plan": MemoryType.PLAN,
            }
            mem_type = type_map.get(memory_type.lower(), MemoryType.CONTEXT)

            embedding = None
            if FOUNDRY_PROJECT_ENDPOINT:
                try:
                    embedding = get_embedding(content)
                except Exception:
                    pass

            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                content=content,
                memory_type=mem_type,
                embedding=embedding,
                session_id=session_id,
            )
            entry_id = await short_term_memory.store(entry)
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"stored": True, "memory_id": entry_id})}])

        elif tool_name == "recall_memory":
            if not short_term_memory:
                return MCPToolResult(content=[{"type": "text", "text": "Memory provider not configured"}], isError=True)

            query = arguments.get("query", "")
            session_id = arguments.get("session_id", "")
            limit = arguments.get("limit", 5)

            query_embedding = None
            if FOUNDRY_PROJECT_ENDPOINT:
                try:
                    query_embedding = get_embedding(query)
                except Exception:
                    pass

            results = await short_term_memory.search(
                query=query, session_id=session_id, limit=limit, embedding=query_embedding
            )
            memories = [{"content": r.entry.content, "score": r.score, "type": r.entry.memory_type.value} for r in results]
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"memories": memories}, indent=2)}])

        # ── Agent Lightning Tools ──
        elif tool_name == "lightning_list_episodes":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            limit = arguments.get("limit", 20)
            start_date = arguments.get("start_date")
            end_date = arguments.get("end_date")

            episodes = rl_ledger.query_episodes(agent_id=agent_id, start_date=start_date, end_date=end_date, limit=limit)
            episodes_data = []
            for ep in episodes:
                episodes_data.append({
                    "id": ep.id,
                    "agent_id": ep.agent_id,
                    "user_input": ep.user_input[:200] + "..." if len(ep.user_input) > 200 else ep.user_input,
                    "assistant_output": ep.assistant_output[:200] + "..." if len(ep.assistant_output) > 200 else ep.assistant_output,
                    "tool_calls_count": len(ep.tool_calls),
                    "model_deployment": ep.model_deployment,
                    "request_latency_ms": ep.request_latency_ms,
                    "created_at": ep.created_at,
                })

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"agent_id": agent_id, "episodes_found": len(episodes_data), "episodes": episodes_data}, indent=2)}])

        elif tool_name == "lightning_get_episode":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            episode_id = arguments.get("episode_id")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID

            if not episode_id:
                return MCPToolResult(content=[{"type": "text", "text": "No episode_id provided"}], isError=True)

            episode = rl_ledger.get_episode(episode_id, agent_id)
            if not episode:
                return MCPToolResult(content=[{"type": "text", "text": f"Episode {episode_id} not found"}], isError=True)

            tool_calls_data = []
            for tc in episode.tool_calls:
                tool_calls_data.append({
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result[:500] + "..." if tc.result and len(tc.result) > 500 else tc.result,
                    "duration_ms": tc.duration_ms,
                    "error": tc.error,
                })

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                "id": episode.id, "agent_id": episode.agent_id, "user_input": episode.user_input,
                "assistant_output": episode.assistant_output, "tool_calls": tool_calls_data,
                "model_deployment": episode.model_deployment, "request_latency_ms": episode.request_latency_ms,
                "created_at": episode.created_at,
            }, indent=2)}])

        elif tool_name == "lightning_assign_reward":
            if not LIGHTNING_AVAILABLE or not reward_writer:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            episode_id = arguments.get("episode_id")
            reward_value = arguments.get("reward_value")
            reward_source = arguments.get("reward_source", "human_approval")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            rubric = arguments.get("rubric")
            evaluator = arguments.get("evaluator")
            comments = arguments.get("comments")

            if not episode_id or reward_value is None:
                return MCPToolResult(content=[{"type": "text", "text": "episode_id and reward_value are required"}], isError=True)

            source_map = {
                "human_approval": RewardSource.HUMAN_APPROVAL,
                "eval_score": RewardSource.EVAL_SCORE,
                "test_result": RewardSource.TEST_RESULT,
                "safety_check": RewardSource.SAFETY_CHECK,
            }
            source = source_map.get(reward_source.lower(), RewardSource.EVAL_SCORE)

            reward = reward_writer.record_reward(
                episode_id=episode_id, agent_id=agent_id, source=source, value=reward_value,
                rubric=rubric, evaluator=evaluator, metadata={"comments": comments} if comments else {},
            )

            if reward:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "reward_id": reward.id, "episode_id": episode_id,
                    "value": reward.value, "source": source.value, "created_at": reward.created_at,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "Failed to store reward"}], isError=True)

        elif tool_name == "lightning_list_rewards":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            episode_id = arguments.get("episode_id")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            limit = arguments.get("limit", 50)

            rewards = rl_ledger.query_rewards(agent_id=agent_id, episode_id=episode_id, limit=limit)
            rewards_data = [{"id": r.id, "episode_id": r.episode_id, "source": r.source.value, "value": r.value, "rubric": r.rubric, "evaluator": r.evaluator, "created_at": r.created_at} for r in rewards]

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"agent_id": agent_id, "rewards_found": len(rewards_data), "rewards": rewards_data}, indent=2)}])

        elif tool_name == "lightning_build_dataset":
            if not LIGHTNING_AVAILABLE or not dataset_builder:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            name = arguments.get("name")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            description = arguments.get("description")
            min_reward = arguments.get("min_reward", 0.5)

            if not name:
                return MCPToolResult(content=[{"type": "text", "text": "name is required"}], isError=True)

            dataset = dataset_builder.build_dataset(agent_id=agent_id, name=name, description=description, min_reward=min_reward)

            if dataset:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "dataset_id": dataset.id, "name": dataset.name,
                    "training_count": dataset.training_count, "validation_count": dataset.validation_count,
                    "episode_count": len(dataset.episode_ids), "local_path": dataset.local_path, "created_at": dataset.created_at,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "No qualifying episodes found for dataset"}], isError=True)

        elif tool_name == "lightning_list_datasets":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            limit = arguments.get("limit", 20)
            datasets = rl_ledger.list_datasets(agent_id=agent_id, limit=limit)
            datasets_data = [{"id": ds.id, "name": ds.name, "training_count": ds.training_count, "validation_count": ds.validation_count, "episode_count": len(ds.episode_ids), "created_at": ds.created_at} for ds in datasets]

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"agent_id": agent_id, "datasets_found": len(datasets_data), "datasets": datasets_data}, indent=2)}])

        elif tool_name == "lightning_start_training":
            if not LIGHTNING_AVAILABLE or not training_runner:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            dataset_id = arguments.get("dataset_id")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            base_model = arguments.get("base_model")
            n_epochs = arguments.get("n_epochs")

            if not dataset_id:
                return MCPToolResult(content=[{"type": "text", "text": "dataset_id is required"}], isError=True)

            hyperparams = {}
            if n_epochs:
                hyperparams["n_epochs"] = n_epochs

            run = training_runner.start_training(dataset_id=dataset_id, agent_id=agent_id, base_model=base_model, hyperparameters=hyperparams if hyperparams else None)

            if run:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "training_run_id": run.id, "dataset_id": run.dataset_id,
                    "base_model": run.base_model, "status": run.status.value, "aoai_job_id": run.aoai_job_id, "created_at": run.created_at,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "Failed to start training"}], isError=True)

        elif tool_name == "lightning_get_training_status":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            training_run_id = arguments.get("training_run_id")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID

            if not training_run_id:
                return MCPToolResult(content=[{"type": "text", "text": "training_run_id is required"}], isError=True)

            if training_runner:
                run = training_runner.check_status(training_run_id, agent_id)
            else:
                run = rl_ledger.get_training_run(training_run_id, agent_id)

            if not run:
                return MCPToolResult(content=[{"type": "text", "text": f"Training run {training_run_id} not found"}], isError=True)

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                "id": run.id, "dataset_id": run.dataset_id, "base_model": run.base_model,
                "tuned_model_name": run.tuned_model_name, "status": run.status.value,
                "metrics": run.metrics, "aoai_job_id": run.aoai_job_id,
                "error_message": run.error_message, "started_at": run.started_at, "completed_at": run.completed_at,
            }, indent=2)}])

        elif tool_name == "lightning_list_training_runs":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            limit = arguments.get("limit", 20)
            runs = rl_ledger.list_training_runs(agent_id=agent_id, limit=limit)
            runs_data = [{"id": r.id, "dataset_id": r.dataset_id, "base_model": r.base_model, "tuned_model_name": r.tuned_model_name, "status": r.status.value, "started_at": r.started_at, "completed_at": r.completed_at} for r in runs]

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"agent_id": agent_id, "runs_found": len(runs_data), "training_runs": runs_data}, indent=2)}])

        elif tool_name == "lightning_promote_deployment":
            if not LIGHTNING_AVAILABLE or not deployment_registry:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            training_run_id = arguments.get("training_run_id")
            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            promoted_by = arguments.get("promoted_by")

            if not training_run_id:
                return MCPToolResult(content=[{"type": "text", "text": "training_run_id is required"}], isError=True)

            deployment = deployment_registry.promote(agent_id=agent_id, training_run_id=training_run_id, promoted_by=promoted_by)

            if deployment:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "deployment_id": deployment.id, "tuned_model_name": deployment.tuned_model_name,
                    "is_active": deployment.is_active, "promoted_at": deployment.promoted_at,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "Failed to promote deployment"}], isError=True)

        elif tool_name == "lightning_get_active_deployment":
            if not LIGHTNING_AVAILABLE or not deployment_registry:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            deployment = deployment_registry.get_active_deployment(agent_id)

            if deployment:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "has_active_deployment": True, "deployment_id": deployment.id,
                    "tuned_model_name": deployment.tuned_model_name, "promoted_at": deployment.promoted_at,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                "has_active_deployment": False, "message": "No active tuned model. Using base model.",
                "base_model": FOUNDRY_MODEL_DEPLOYMENT_NAME,
            }, indent=2)}])

        elif tool_name == "lightning_list_deployments":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            limit = arguments.get("limit", 20)
            deployments = rl_ledger.list_deployments(agent_id=agent_id, limit=limit)
            deployments_data = [{"id": dep.id, "tuned_model_name": dep.tuned_model_name, "is_active": dep.is_active, "promoted_at": dep.promoted_at, "rollback_from": dep.rollback_from} for dep in deployments]

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({"agent_id": agent_id, "deployments_found": len(deployments_data), "deployments": deployments_data}, indent=2)}])

        elif tool_name == "lightning_rollback_deployment":
            if not LIGHTNING_AVAILABLE or not deployment_registry:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            target_deployment_id = arguments.get("target_deployment_id")
            reason = arguments.get("reason")
            rolled_back_by = arguments.get("rolled_back_by")

            deployment = deployment_registry.rollback(agent_id=agent_id, target_deployment_id=target_deployment_id, reason=reason, rolled_back_by=rolled_back_by)

            if deployment:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "deployment_id": deployment.id, "tuned_model_name": deployment.tuned_model_name,
                    "is_active": deployment.is_active, "rollback_reason": reason,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "No previous deployment to roll back to"}], isError=True)

        elif tool_name == "lightning_deactivate_deployment":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID
            reason = arguments.get("reason")
            success = rl_ledger.deactivate_deployment(agent_id=agent_id)

            if success:
                return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                    "success": True, "message": "Tuned model deactivated. Now using base model.",
                    "base_model": FOUNDRY_MODEL_DEPLOYMENT_NAME, "reason": reason,
                }, indent=2)}])
            return MCPToolResult(content=[{"type": "text", "text": "Failed to deactivate deployment"}], isError=True)

        elif tool_name == "lightning_get_stats":
            if not LIGHTNING_AVAILABLE or not rl_ledger:
                return MCPToolResult(content=[{"type": "text", "text": "Agent Lightning not available"}], isError=True)

            agent_id = arguments.get("agent_id") or LIGHTNING_AGENT_ID

            episodes = rl_ledger.query_episodes(agent_id=agent_id, limit=1000)
            rewards = rl_ledger.query_rewards(agent_id=agent_id, limit=1000)
            datasets = rl_ledger.list_datasets(agent_id=agent_id, limit=100)
            runs = rl_ledger.list_training_runs(agent_id=agent_id, limit=100)
            deployments = rl_ledger.list_deployments(agent_id=agent_id, limit=100)

            active_deployment = deployment_registry.get_active_deployment(agent_id) if deployment_registry else None
            reward_values = [r.value for r in rewards]
            avg_reward = sum(reward_values) / len(reward_values) if reward_values else 0

            status_counts = {}
            for run in runs:
                status = run.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return MCPToolResult(content=[{"type": "text", "text": json.dumps({
                "agent_id": agent_id, "lightning_enabled": ENABLE_LIGHTNING_CAPTURE, "use_tuned_model": USE_TUNED_MODEL,
                "statistics": {
                    "total_episodes": len(episodes), "total_rewards": len(rewards),
                    "average_reward": round(avg_reward, 3), "total_datasets": len(datasets),
                    "total_training_runs": len(runs), "training_run_status": status_counts,
                    "total_deployments": len(deployments),
                },
                "active_deployment": {
                    "has_active": active_deployment is not None,
                    "model_name": active_deployment.tuned_model_name if active_deployment else None,
                } if active_deployment else {"has_active": False},
                "current_model": get_model_deployment(), "base_model": FOUNDRY_MODEL_DEPLOYMENT_NAME,
            }, indent=2)}])

        else:
            return MCPToolResult(content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}], isError=True)

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return MCPToolResult(content=[{"type": "text", "text": f"Error: {str(e)}"}], isError=True)


# =========================================
# FastAPI Endpoints
# =========================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/runtime/webhooks/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for MCP protocol."""
    session_id = str(uuid.uuid4())
    logger.info(f"New SSE session established: {session_id}")

    sessions[session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "message_queue": asyncio.Queue(),
    }

    async def event_generator():
        try:
            message_url = f"message?sessionId={session_id}"
            yield f"data: {message_url}\n\n"

            while True:
                if session_id not in sessions:
                    break
                try:
                    message = await asyncio.wait_for(
                        sessions[session_id]["message_queue"].get(), timeout=30.0
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
    """Message endpoint for MCP protocol – JSON-RPC 2.0 requests."""
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
                content={"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request_id},
            )

        # Handle initialize
        if method == "initialize":
            return JSONResponse(content={
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "um-agent", "version": "1.0.0"},
                },
                "id": request_id,
            })

        # Handle tools/list
        elif method == "tools/list":
            tools_list = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in TOOLS]
            return JSONResponse(content={"jsonrpc": "2.0", "result": {"tools": tools_list}, "id": request_id})

        # Handle tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await execute_tool(tool_name, arguments)
            return JSONResponse(content={"jsonrpc": "2.0", "result": asdict(result), "id": request_id})

        else:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Method not found: {method}"}, "id": request_id},
            )

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return JSONResponse(
            status_code=500,
            content={"jsonrpc": "2.0", "error": {"code": -32603, "message": f"Internal error: {str(e)}"}, "id": body.get("id") if "body" in dir() else None},
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "UM Agent MCP Server",
        "version": "1.0.0",
        "domain": "utilization_management",
        "endpoints": {
            "sse": "/runtime/webhooks/mcp/sse",
            "message": "/runtime/webhooks/mcp/message",
            "health": "/health",
        },
    }


@app.on_event("startup")
async def startup_event():
    """Initialize memory providers on startup."""
    if short_term_memory and FOUNDRY_PROJECT_ENDPOINT:
        short_term_memory.set_embedding_function(get_embedding)
        logger.info("Memory provider embedding function configured")

    if composite_memory:
        health = await composite_memory.health_check()
        for provider, is_healthy in health.items():
            status = "healthy" if is_healthy else "unhealthy"
            logger.info(f"Memory provider '{provider}': {status}")
