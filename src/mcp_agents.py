"""
FastAPI MCP Agents
Implements Model Context Protocol (MCP) with SSE support
Enhanced with Microsoft Agent Framework for AI agent capabilities
Integrated with CosmosDB for task and plan storage with semantic reasoning
Features Memory Provider abstraction for short-term (CosmosDB), long-term (AI Search), and facts (Fabric IQ) memory
"""

import json
import logging
import asyncio
import uuid
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
import os

# Microsoft Agent Framework imports
from agent_framework import ai_function, AIFunction
from agent_framework.azure import AzureAIAgentClient

# Memory Provider imports
from memory import (
    ShortTermMemory, MemoryEntry, MemoryType, CompositeMemory, LongTermMemory,
    # Fabric IQ Facts Memory
    FactsMemory, Fact, FactSearchResult, OntologyEntity, EntityType, RelationshipType,
    # Domain ontology data generators
    CustomerDataGenerator, CustomerProfile, CustomerSegment, ChurnRiskLevel,
    PipelineDataGenerator, Pipeline, PipelineRun, PipelineStatus,
    UserAccessDataGenerator, User, AuthEvent, AuthEventType,
)

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server for AI Agents with Semantic Reasoning",
    version="1.0.0"
)

# Azure Storage configuration
STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# CosmosDB configuration
COSMOSDB_ENDPOINT = os.getenv("COSMOSDB_ENDPOINT", "")
COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME", "mcpdb")
COSMOSDB_TASKS_CONTAINER = "tasks"
COSMOSDB_PLANS_CONTAINER = "plans"

# Initialize storage client
if STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
elif STORAGE_ACCOUNT_URL:
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
else:
    logger.warning("No storage configuration found - snippet storage will not work")
    blob_service_client = None

# Initialize CosmosDB client
cosmos_client = None
cosmos_database = None
cosmos_tasks_container = None
cosmos_plans_container = None

if COSMOSDB_ENDPOINT:
    try:
        credential = DefaultAzureCredential()
        cosmos_client = CosmosClient(COSMOSDB_ENDPOINT, credential=credential)
        cosmos_database = cosmos_client.get_database_client(COSMOSDB_DATABASE_NAME)
        cosmos_tasks_container = cosmos_database.get_container_client(COSMOSDB_TASKS_CONTAINER)
        cosmos_plans_container = cosmos_database.get_container_client(COSMOSDB_PLANS_CONTAINER)
        logger.info("CosmosDB client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CosmosDB client: {e}")
else:
    logger.warning("COSMOSDB_ENDPOINT not configured - task storage will not work")

# Initialize Memory Providers
short_term_memory: Optional[ShortTermMemory] = None
composite_memory: Optional[CompositeMemory] = None

if COSMOSDB_ENDPOINT:
    try:
        short_term_memory = ShortTermMemory(
            endpoint=COSMOSDB_ENDPOINT,
            database_name=COSMOSDB_DATABASE_NAME,
            container_name="short_term_memory",
            default_ttl=3600,  # 1 hour default TTL
        )
        
        # Create composite memory (long-term will be added later with AI Search)
        composite_memory = CompositeMemory(
            short_term=short_term_memory,
            long_term=None,  # Will be AI Search / FoundryIQ
        )
        
        logger.info("Memory providers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory providers: {e}")
else:
    logger.warning("COSMOSDB_ENDPOINT not configured - memory providers will not work")

SNIPPETS_CONTAINER = "snippets"

# In-memory session storage (replace with Redis for production)
sessions: Dict[str, Dict[str, Any]] = {}

# Microsoft Agent Framework configuration
FOUNDRY_PROJECT_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
FOUNDRY_MODEL_DEPLOYMENT_NAME = os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-5.2-chat")
EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-large")

# Azure AI Search configuration for long-term memory
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "task-instructions")

# Microsoft Fabric IQ configuration for Facts Memory
FABRIC_ENABLED = os.getenv("FABRIC_ENABLED", "false").lower() == "true"
FABRIC_ENDPOINT = os.getenv("FABRIC_ENDPOINT", "")
FABRIC_WORKSPACE_ID = os.getenv("FABRIC_WORKSPACE_ID", "")
FABRIC_ONTOLOGY_NAME = os.getenv("FABRIC_ONTOLOGY_NAME", "agent-ontology")
# OneLake configuration for ontology storage (when Fabric is enabled)
FABRIC_ONELAKE_DFS_ENDPOINT = os.getenv("FABRIC_ONELAKE_DFS_ENDPOINT", "https://onelake.dfs.fabric.microsoft.com")
FABRIC_ONELAKE_BLOB_ENDPOINT = os.getenv("FABRIC_ONELAKE_BLOB_ENDPOINT", "https://onelake.blob.fabric.microsoft.com")
FABRIC_LAKEHOUSE_NAME = os.getenv("FABRIC_LAKEHOUSE_NAME", "mcpontologies")
FABRIC_ONTOLOGY_PATH = os.getenv("FABRIC_ONTOLOGY_PATH", "Files/ontology")
# Ontology storage configuration (when Fabric is disabled - uses Azure Blob Storage)
ONTOLOGY_CONTAINER_NAME = os.getenv("ONTOLOGY_CONTAINER_NAME", "ontologies")
AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")

# AI Search Long-Term Memory will be initialized after helper functions are defined
long_term_memory: Optional[LongTermMemory] = None

# Fabric IQ Facts Memory for ontology-grounded facts
facts_memory: Optional[FactsMemory] = None


# =========================================
# Embedding and Semantic Reasoning Helpers
# =========================================

def get_embedding(text: str) -> List[float]:
    """
    Generate embeddings for text using Azure AI Foundry's text-embedding-3-large model.
    
    Args:
        text: The text to generate embeddings for
    
    Returns:
        A list of floats representing the embedding vector (3072 dimensions)
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        raise ValueError("Foundry endpoint not configured")
    
    from openai import AzureOpenAI
    
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    
    base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
    
    client = AzureOpenAI(
        azure_endpoint=base_endpoint,
        api_key=token.token,
        api_version="2024-02-15-preview"
    )
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL_DEPLOYMENT_NAME,
        input=text
    )
    
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)
    
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def find_similar_tasks(task_embedding: List[float], threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find similar tasks in CosmosDB using cosine similarity.
    
    Args:
        task_embedding: The embedding vector of the current task
        threshold: Minimum similarity score (0-1)
        limit: Maximum number of similar tasks to return
    
    Returns:
        List of similar tasks with their similarity scores
    """
    if not cosmos_tasks_container:
        return []
    
    try:
        # Query all tasks with embeddings
        query = "SELECT c.id, c.task, c.intent, c.embedding, c.created_at FROM c WHERE IS_DEFINED(c.embedding)"
        items = list(cosmos_tasks_container.query_items(query=query, enable_cross_partition_query=True))
        
        similar_tasks = []
        for item in items:
            if 'embedding' in item and item['embedding']:
                similarity = cosine_similarity(task_embedding, item['embedding'])
                if similarity >= threshold:
                    similar_tasks.append({
                        'id': item['id'],
                        'task': item.get('task', ''),
                        'intent': item.get('intent', ''),
                        'similarity': similarity,
                        'created_at': item.get('created_at', '')
                    })
        
        # Sort by similarity descending and limit results
        similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_tasks[:limit]
    
    except Exception as e:
        logger.error(f"Error finding similar tasks: {e}")
        return []


def analyze_intent(task: str) -> str:
    """
    Use the LLM to analyze and categorize the intent of a task.
    
    Args:
        task: The task description in natural language
    
    Returns:
        A string describing the analyzed intent
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        return "unknown"
    
    try:
        from openai import AzureOpenAI
        
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
        
        client = AzureOpenAI(
            azure_endpoint=base_endpoint,
            api_key=token.token,
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=FOUNDRY_MODEL_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a task analyzer. Analyze the given task and provide a brief categorization of its intent. Return only a short phrase describing the primary intent (e.g., 'data analysis', 'code generation', 'information retrieval', 'system configuration')."
                },
                {"role": "user", "content": f"Analyze this task: {task}"}
            ]
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        return "unknown"
    
    except Exception as e:
        logger.error(f"Error analyzing intent: {e}")
        return "unknown"


def generate_plan(task: str, similar_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate a plan of steps to accomplish the task, optionally learning from similar past tasks.
    
    Args:
        task: The task description
        similar_tasks: List of similar tasks for context
    
    Returns:
        List of planned steps
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        return [{"step": 1, "action": "Manual planning required", "description": "Foundry not configured"}]
    
    try:
        from openai import AzureOpenAI
        
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
        
        client = AzureOpenAI(
            azure_endpoint=base_endpoint,
            api_key=token.token,
            api_version="2024-02-15-preview"
        )
        
        # Build context from similar tasks
        context = ""
        if similar_tasks:
            context = "\n\nSimilar past tasks for reference:\n"
            for st in similar_tasks[:3]:
                context += f"- {st['task']} (intent: {st['intent']}, similarity: {st['similarity']:.2f})\n"
        
        response = client.chat.completions.create(
            model=FOUNDRY_MODEL_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """You are a task planner. Given a task, generate a structured plan with actionable steps.
Return a JSON array of steps, each with:
- "step": step number (integer)
- "action": brief action title
- "description": detailed description of what to do
- "estimated_effort": low/medium/high

Return ONLY valid JSON array, no markdown or explanation."""
                },
                {"role": "user", "content": f"Create a plan for this task: {task}{context}"}
            ]
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            # Parse JSON from response
            try:
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                return json.loads(content)
            except json.JSONDecodeError:
                return [{"step": 1, "action": "Execute task", "description": content, "estimated_effort": "medium"}]
        
        return [{"step": 1, "action": "Execute task", "description": task, "estimated_effort": "medium"}]
    
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        return [{"step": 1, "action": "Error", "description": str(e), "estimated_effort": "unknown"}]


def generate_plan_with_instructions(
    task: str,
    similar_tasks: List[Dict[str, Any]],
    task_instructions: List[Dict[str, Any]],
    domain_facts: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Generate a plan of steps to accomplish the task using:
    1. Similar past tasks from CosmosDB (short-term memory)
    2. Task instructions from AI Search (long-term memory)
    3. Domain facts from Fabric IQ (ontology-grounded facts memory)
    
    Args:
        task: The task description
        similar_tasks: List of similar tasks from CosmosDB
        task_instructions: List of task instructions from AI Search
        domain_facts: List of relevant facts from Fabric IQ ontology
    
    Returns:
        List of planned steps
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        return [{"step": 1, "action": "Manual planning required", "description": "Foundry not configured"}]
    
    try:
        from openai import AzureOpenAI
        
        credential = DefaultAzureCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
        
        client = AzureOpenAI(
            azure_endpoint=base_endpoint,
            api_key=token.token,
            api_version="2024-02-15-preview"
        )
        
        # Build context from similar tasks (short-term memory)
        context = ""
        if similar_tasks:
            context = "\n\n## Similar Past Tasks (Short-Term Memory):\n"
            for st in similar_tasks[:3]:
                context += f"- {st['task']} (intent: {st['intent']}, similarity: {st['similarity']:.2f})\n"
        
        # Build context from task instructions (long-term memory from AI Search)
        if task_instructions:
            context += "\n\n## Task Instructions (Long-Term Memory):\n"
            for ti in task_instructions[:2]:
                context += f"\n### {ti.get('title', 'Untitled')} (relevance: {ti.get('score', 0):.2f})\n"
                context += f"Category: {ti.get('category', 'N/A')}\n"
                context += f"Description: {ti.get('description', 'N/A')}\n"
                
                # Include reference steps if available
                ref_steps = ti.get('steps', [])
                if ref_steps:
                    context += "Reference Steps:\n"
                    for step in ref_steps[:5]:  # Limit to first 5 steps
                        context += f"  {step.get('step', '?')}. {step.get('action', 'N/A')}: {step.get('description', 'N/A')[:100]}...\n"
                
                # Include content excerpt
                content_excerpt = ti.get('content_excerpt', '')
                if content_excerpt:
                    context += f"\nKey Information:\n{content_excerpt[:500]}...\n"
        
        # Build context from domain facts (Fabric IQ ontology-grounded facts)
        if domain_facts:
            context += "\n\n## Domain Facts (Fabric IQ Ontology):\n"
            context += "The following facts are derived from the knowledge graph and provide grounded context:\n"
            for fact in domain_facts[:5]:
                context += f"\n### {fact.get('domain', 'unknown').upper()} Domain Fact\n"
                context += f"- Statement: {fact.get('statement', 'N/A')}\n"
                context += f"- Confidence: {fact.get('confidence', 0):.0%}\n"
                context += f"- Type: {fact.get('fact_type', 'N/A')}\n"
                # Include relevant context from the fact
                fact_context = fact.get('context', {})
                if fact_context:
                    context += "- Key Metrics:\n"
                    for key, value in list(fact_context.items())[:5]:
                        context += f"    - {key}: {value}\n"
        
        response = client.chat.completions.create(
            model=FOUNDRY_MODEL_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert task planner with access to:
1. Short-term memory (similar past tasks)
2. Long-term memory (detailed task instructions)
3. Facts memory (ontology-grounded domain facts from Fabric IQ)

Use ALL provided context to generate a highly specific, actionable plan tailored to the task.
When domain facts are available, incorporate the specific metrics and insights into your planning.
When task instructions are available, leverage the reference steps and key information.

Return a JSON array of steps, each with:
- "step": step number (integer)
- "action": brief action title
- "description": detailed description of what to do
- "estimated_effort": low/medium/high
- "source": "original" if new, "adapted" if based on instructions, "fact-grounded" if based on domain facts

Return ONLY valid JSON array, no markdown or explanation."""
                },
                {"role": "user", "content": f"Create a detailed plan for this task: {task}{context}"}
            ]
        )
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            try:
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                return json.loads(content)
            except json.JSONDecodeError:
                return [{"step": 1, "action": "Execute task", "description": content, "estimated_effort": "medium", "source": "original"}]
        
        return [{"step": 1, "action": "Execute task", "description": task, "estimated_effort": "medium", "source": "original"}]
    
    except Exception as e:
        logger.error(f"Error generating plan with instructions: {e}")
        # Fallback to basic plan generation
        return generate_plan(task, similar_tasks)


# =========================================
# Initialize AI Search Long-Term Memory
# (After helper functions are defined)
# =========================================

def _initialize_long_term_memory():
    """Initialize AI Search long-term memory with embedding function."""
    global long_term_memory
    
    if AZURE_SEARCH_ENDPOINT:
        try:
            long_term_memory = LongTermMemory(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=AZURE_SEARCH_INDEX_NAME,
                foundry_endpoint=FOUNDRY_PROJECT_ENDPOINT,
            )
            # Set embedding function for the long-term memory
            if FOUNDRY_PROJECT_ENDPOINT:
                long_term_memory.set_embedding_function(get_embedding)
            
            # Update composite memory with long-term if it exists
            if composite_memory:
                composite_memory._long_term = long_term_memory
            
            logger.info(f"AI Search Long-Term Memory initialized: {AZURE_SEARCH_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize AI Search long-term memory: {e}")
    else:
        logger.warning("AZURE_SEARCH_ENDPOINT not configured - long-term memory will not work")


def _initialize_facts_memory():
    """
    Initialize Facts Memory with ontology-grounded facts.
    
    Uses Azure Blob Storage by default (when FABRIC_ENABLED=false).
    Uses Fabric IQ when FABRIC_ENABLED=true.
    
    Loads sample data for Customer, DevOps, and User Management domains.
    """
    global facts_memory
    
    try:
        # Initialize with Blob Storage (default) or Fabric IQ mode
        facts_memory = FactsMemory(
            storage_account_url=AZURE_STORAGE_ACCOUNT_URL,
            ontology_container=ONTOLOGY_CONTAINER_NAME,
            fabric_enabled=FABRIC_ENABLED,
            fabric_endpoint=FABRIC_ENDPOINT,
            workspace_id=FABRIC_WORKSPACE_ID,
            ontology_name=FABRIC_ONTOLOGY_NAME,
        )
        
        # Set embedding function if available
        if FOUNDRY_PROJECT_ENDPOINT:
            facts_memory.set_embedding_function(get_embedding)
        
        mode = "Fabric IQ" if FABRIC_ENABLED else "Azure Blob Storage"
        logger.info(f"Facts Memory initialized: ontology={FABRIC_ONTOLOGY_NAME}, mode={mode}")
        
        # Load ontologies from storage (if not using in-memory sample data)
        async def load_from_storage():
            if not FABRIC_ENABLED and AZURE_STORAGE_ACCOUNT_URL:
                # Try to load ontologies from Blob Storage
                loaded = await facts_memory.load_all_ontologies()
                if loaded > 0:
                    logger.info(f"Loaded {loaded} ontologies from Azure Blob Storage")
                    return
            # Fallback to sample data if no ontologies loaded from storage
            await _load_sample_ontology_data()
        
        asyncio.get_event_loop().run_until_complete(load_from_storage())
        
    except Exception as e:
        logger.error(f"Failed to initialize Facts Memory: {e}")


async def _load_sample_ontology_data():
    """Load sample ontology data for all three domains."""
    if not facts_memory:
        return
    
    logger.info("Loading sample ontology data for Fabric IQ...")
    
    # =========================================
    # 1. Customer Churn Analysis Domain
    # =========================================
    customers = CustomerDataGenerator.generate_customers(count=25)
    for customer in customers:
        entity = OntologyEntity(
            id=customer.customer_id,
            entity_type=EntityType.CUSTOMER,
            properties=customer.to_dict(),
        )
        await facts_memory.store_entity(entity)
        
        # Derive facts for high-risk customers
        if customer.churn_risk > 0.5:
            fact = Fact(
                id=f"fact-churn-{customer.customer_id}",
                fact_type="prediction",
                domain="customer",
                statement=f"Customer '{customer.name}' ({customer.segment.value} segment) has {customer.churn_risk:.0%} churn risk. "
                          f"Key indicators: {customer.days_since_last_login} days since last login, "
                          f"feature usage score of {customer.feature_usage_score:.0f}/100, "
                          f"NPS score of {customer.nps_score}.",
                confidence=customer.churn_risk,
                evidence=[customer.customer_id],
                context={
                    "segment": customer.segment.value,
                    "risk_level": customer.risk_level.value,
                    "tenure_months": customer.tenure_months,
                    "monthly_spend": customer.monthly_spend,
                },
            )
            await facts_memory.store_fact(fact)
    
    logger.info(f"Loaded {len(customers)} customer entities with churn analysis facts")
    
    # =========================================
    # 2. CI/CD Pipeline Domain
    # =========================================
    pipelines = PipelineDataGenerator.generate_pipelines(count=6)
    total_runs = 0
    
    for pipeline in pipelines:
        entity = OntologyEntity(
            id=pipeline.pipeline_id,
            entity_type=EntityType.PIPELINE,
            properties=pipeline.to_dict(),
        )
        await facts_memory.store_entity(entity)
        
        # Generate pipeline runs with successes and failures
        runs = PipelineDataGenerator.generate_pipeline_runs(pipeline, count=20)
        total_runs += len(runs)
        
        success_runs = [r for r in runs if r.status == PipelineStatus.SUCCESS]
        failed_runs = [r for r in runs if r.status == PipelineStatus.FAILURE]
        
        for run in runs:
            run_entity = OntologyEntity(
                id=run.run_id,
                entity_type=EntityType.PIPELINE_RUN,
                properties=run.to_dict(),
            )
            await facts_memory.store_entity(run_entity)
        
        # Create pipeline health facts
        success_rate = len(success_runs) / len(runs) if runs else 0
        fact = Fact(
            id=f"fact-pipeline-{pipeline.pipeline_id}",
            fact_type="observation",
            domain="devops",
            statement=f"Pipeline '{pipeline.name}' for {pipeline.service_name} has {success_rate:.0%} success rate "
                      f"over {len(runs)} recent runs. Target cluster: {pipeline.target_cluster}. "
                      f"{len(failed_runs)} failures detected.",
            confidence=0.95,
            evidence=[pipeline.pipeline_id] + [r.run_id for r in runs[:5]],
            context={
                "success_rate": success_rate,
                "total_runs": len(runs),
                "failures": len(failed_runs),
                "avg_duration": pipeline.avg_duration_seconds,
                "service": pipeline.service_name,
            },
        )
        await facts_memory.store_fact(fact)
        
        # Create facts for significant failures
        for run in failed_runs[:3]:
            failure_fact = Fact(
                id=f"fact-failure-{run.run_id}",
                fact_type="observation",
                domain="devops",
                statement=f"Pipeline run {run.run_id} failed at stage '{run.failure_stage}' with error: {run.failure_message}. "
                          f"Category: {run.failure_category}. Triggered by: {run.triggered_by}.",
                confidence=1.0,
                evidence=[run.run_id, pipeline.pipeline_id],
                context={
                    "failure_category": run.failure_category,
                    "failure_stage": run.failure_stage,
                    "commit_sha": run.commit_sha,
                    "duration_seconds": run.duration_seconds,
                },
            )
            await facts_memory.store_fact(failure_fact)
    
    logger.info(f"Loaded {len(pipelines)} pipelines with {total_runs} execution runs")
    
    # =========================================
    # 3. User Management Domain
    # =========================================
    users = UserAccessDataGenerator.generate_users(count=15)
    total_auth_events = 0
    
    for user in users:
        entity = OntologyEntity(
            id=user.user_id,
            entity_type=EntityType.USER,
            properties=user.to_dict(),
        )
        await facts_memory.store_entity(entity)
        
        # Generate auth events
        auth_events = UserAccessDataGenerator.generate_auth_events(user, count=30)
        total_auth_events += len(auth_events)
        
        for event in auth_events:
            event_entity = OntologyEntity(
                id=event.event_id,
                entity_type=EntityType.AUTH_EVENT,
                properties=event.to_dict(),
            )
            await facts_memory.store_entity(event_entity)
        
        # Create user activity facts
        login_successes = len([e for e in auth_events if e.event_type == AuthEventType.LOGIN_SUCCESS])
        login_failures = len([e for e in auth_events if e.event_type == AuthEventType.LOGIN_FAILURE])
        high_risk_events = len([e for e in auth_events if e.risk_score > 0.5])
        
        fact = Fact(
            id=f"fact-user-{user.user_id}",
            fact_type="observation",
            domain="user_management",
            statement=f"User '{user.username}' ({', '.join(user.roles)}) has {login_successes} successful logins "
                      f"and {login_failures} failed attempts. MFA enabled: {user.mfa_enabled}. "
                      f"Status: {user.status.value}. {high_risk_events} high-risk authentication events detected.",
            confidence=0.9,
            evidence=[user.user_id] + [e.event_id for e in auth_events[:5]],
            context={
                "roles": user.roles,
                "mfa_enabled": user.mfa_enabled,
                "status": user.status.value,
                "login_successes": login_successes,
                "login_failures": login_failures,
                "high_risk_events": high_risk_events,
            },
        )
        await facts_memory.store_fact(fact)
        
        # Flag suspicious users
        if login_failures > 5 or high_risk_events > 3:
            security_fact = Fact(
                id=f"fact-security-{user.user_id}",
                fact_type="derived",
                domain="user_management",
                statement=f"SECURITY ALERT: User '{user.username}' shows suspicious activity pattern. "
                          f"{login_failures} failed login attempts, {high_risk_events} high-risk events. "
                          f"Recommend review of account activity.",
                confidence=0.85,
                evidence=[user.user_id],
                context={
                    "alert_type": "suspicious_activity",
                    "failed_logins": login_failures,
                    "high_risk_events": high_risk_events,
                },
            )
            await facts_memory.store_fact(security_fact)
    
    logger.info(f"Loaded {len(users)} users with {total_auth_events} auth events")
    
    # Log final statistics
    stats = facts_memory.get_stats()
    logger.info(f"Facts Memory loaded: {stats['total_entities']} entities, {stats['total_facts']} facts")


# Initialize long-term memory now that helper functions are available
_initialize_long_term_memory()

# Initialize facts memory with sample ontology data
_initialize_facts_memory()


# Define Agent Framework tools using @ai_function decorator
@ai_function
def hello_mcp_tool() -> str:
    """Hello world MCP tool that returns a greeting message."""
    return "Hello I am MCPTool!"


@ai_function
def get_snippet_tool(snippetname: str) -> str:
    """
    Retrieve a snippet by name from Azure Blob Storage.
    
    Args:
        snippetname: The name of the snippet to retrieve
    
    Returns:
        The content of the snippet
    """
    if not blob_service_client:
        return "Error: Storage not configured"
    
    try:
        blob_client = blob_service_client.get_blob_client(
            container=SNIPPETS_CONTAINER,
            blob=f"{snippetname}.json"
        )
        blob_data = blob_client.download_blob().readall()
        return blob_data.decode('utf-8')
    except Exception as e:
        logger.error(f"Error retrieving snippet: {e}")
        return f"Error retrieving snippet: {str(e)}"


@ai_function
def save_snippet_tool(snippetname: str, snippet: str) -> str:
    """
    Save a snippet with a name to Azure Blob Storage.
    
    Args:
        snippetname: The name of the snippet
        snippet: The content of the snippet
    
    Returns:
        Success or error message
    """
    if not blob_service_client:
        return "Error: Storage not configured"
    
    try:
        blob_client = blob_service_client.get_blob_client(
            container=SNIPPETS_CONTAINER,
            blob=f"{snippetname}.json"
        )
        blob_client.upload_blob(snippet.encode('utf-8'), overwrite=True)
        return f"Snippet '{snippetname}' saved successfully"
    except Exception as e:
        logger.error(f"Error saving snippet: {e}")
        return f"Error saving snippet: {str(e)}"


@ai_function
def ask_foundry_tool(question: str) -> str:
    """
    Ask a question and get an answer using the Azure AI Foundry model.
    
    Args:
        question: The question to ask the AI model
    
    Returns:
        The AI model's response to the question
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        return "Error: Foundry endpoint not configured"
    
    try:
        from openai import AzureOpenAI
        
        credential = DefaultAzureCredential()
        # Get a token for Azure Cognitive Services
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        
        # Extract the base endpoint (remove /api/projects/proj-default if present)
        # Use the services.ai.azure.com endpoint directly
        base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
        
        client = AzureOpenAI(
            azure_endpoint=base_endpoint,
            api_key=token.token,
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=FOUNDRY_MODEL_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": question}]
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "No response generated"
    except Exception as e:
        logger.error(f"Error calling Foundry model: {e}")
        return f"Error calling Foundry model: {str(e)}"


@ai_function
def next_best_action_tool(task: str) -> str:
    """
    Analyze a task using semantic reasoning with three memory layers:
    1. Short-term memory (CosmosDB) - finds similar past tasks using cosine similarity
    2. Long-term memory (Foundry IQ) - retrieves relevant task instructions
    3. Facts memory (Fabric IQ) - queries domain facts from ontologies
    
    The planning is grounded in domain knowledge from Fabric IQ ontologies:
    - Customer domain: churn predictions, segment analysis, risk assessments
    - DevOps domain: pipeline health, deployment status, failure patterns
    - User Management domain: authentication patterns, security alerts
    
    Args:
        task: The task description in natural language (English sentence)
    
    Returns:
        A JSON response containing task analysis, similar tasks, domain facts, and planned steps
    """
    if not FOUNDRY_PROJECT_ENDPOINT:
        return json.dumps({"error": "Foundry endpoint not configured"})
    
    if not cosmos_tasks_container or not cosmos_plans_container:
        return json.dumps({"error": "CosmosDB not configured"})
    
    try:
        import asyncio
        
        task_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Step 1: Generate embedding for the task
        logger.info(f"Generating embedding for task: {task[:100]}...")
        task_embedding = get_embedding(task)
        
        # Step 2: Analyze intent
        logger.info("Analyzing task intent...")
        intent = analyze_intent(task)
        
        # Step 3: Find similar tasks using cosine similarity (short-term memory from CosmosDB)
        logger.info("Searching for similar past tasks in CosmosDB...")
        similar_tasks = find_similar_tasks(task_embedding, threshold=0.7, limit=5)
        
        # Step 4: Search for task instructions in AI Search long-term memory
        task_instructions = []
        if long_term_memory:
            logger.info("Searching for task instructions in AI Search...")
            try:
                # Run async search in sync context
                loop = asyncio.new_event_loop()
                task_instructions = loop.run_until_complete(
                    long_term_memory.search_task_instructions(
                        task_description=task,
                        limit=3,
                        include_steps=True
                    )
                )
                loop.close()
                logger.info(f"Found {len(task_instructions)} relevant task instructions")
            except Exception as e:
                logger.warning(f"Failed to retrieve task instructions from AI Search: {e}")
        else:
            logger.info("AI Search long-term memory not configured - skipping task instructions lookup")
        
        # Step 5: Search for domain facts in Fabric IQ facts memory
        domain_facts = []
        if facts_memory:
            logger.info("Searching for domain facts in Fabric IQ...")
            try:
                loop = asyncio.new_event_loop()
                # Search for facts relevant to the task across all domains
                fact_results = loop.run_until_complete(
                    facts_memory.search_facts(
                        query=task,
                        domain=None,  # Search all domains
                        limit=5,
                    )
                )
                loop.close()
                
                # Convert fact results to dictionary format for planning
                for result in fact_results:
                    domain_facts.append({
                        'id': result.fact.id,
                        'statement': result.fact.statement,
                        'domain': result.fact.domain,
                        'fact_type': result.fact.fact_type,
                        'confidence': result.fact.confidence,
                        'relevance_score': result.score,
                        'context': result.fact.context,
                        'evidence': result.fact.evidence,
                    })
                logger.info(f"Found {len(domain_facts)} relevant domain facts from Fabric IQ")
            except Exception as e:
                logger.warning(f"Failed to retrieve facts from Fabric IQ: {e}")
        else:
            logger.info("Fabric IQ facts memory not configured - skipping domain facts lookup")
        
        # Step 6: Generate plan based on task, similar past tasks, task instructions, AND domain facts
        logger.info("Generating execution plan with all memory contexts...")
        plan_steps = generate_plan_with_instructions(task, similar_tasks, task_instructions, domain_facts)
        
        # Step 7: Store task in CosmosDB
        task_doc = {
            'id': task_id,
            'task': task,
            'intent': intent,
            'embedding': task_embedding,
            'created_at': timestamp,
            'similar_task_count': len(similar_tasks),
            'task_instructions_found': len(task_instructions),
            'domain_facts_found': len(domain_facts)
        }
        cosmos_tasks_container.upsert_item(task_doc)
        logger.info(f"Task stored in CosmosDB with id: {task_id}")
        
        # Step 8: Store plan in CosmosDB
        plan_doc = {
            'id': str(uuid.uuid4()),
            'taskId': task_id,
            'task': task,
            'intent': intent,
            'steps': plan_steps,
            'similar_tasks_referenced': [{'id': st['id'], 'similarity': st['similarity']} for st in similar_tasks],
            'task_instructions_used': [ti.get('document_id', '') for ti in task_instructions],
            'domain_facts_used': [df.get('id', '') for df in domain_facts],
            'created_at': timestamp,
            'status': 'planned'
        }
        cosmos_plans_container.upsert_item(plan_doc)
        logger.info(f"Plan stored in CosmosDB for task: {task_id}")
        
        # Build response
        response = {
            'task_id': task_id,
            'task': task,
            'intent': intent,
            'analysis': {
                'similar_tasks_found': len(similar_tasks),
                'similar_tasks': [
                    {
                        'task': st['task'],
                        'intent': st['intent'],
                        'similarity_score': round(st['similarity'], 3)
                    }
                    for st in similar_tasks
                ],
                'task_instructions_found': len(task_instructions),
                'task_instructions': [
                    {
                        'title': ti.get('title', ''),
                        'category': ti.get('category', ''),
                        'intent': ti.get('intent', ''),
                        'description': ti.get('description', ''),
                        'relevance_score': round(ti.get('score', 0), 3),
                        'estimated_effort': ti.get('estimated_effort', ''),
                        'reference_steps_count': len(ti.get('steps', []))
                    }
                    for ti in task_instructions
                ],
                'domain_facts_found': len(domain_facts),
                'domain_facts': [
                    {
                        'id': df.get('id', ''),
                        'domain': df.get('domain', ''),
                        'fact_type': df.get('fact_type', ''),
                        'statement': df.get('statement', '')[:200] + '...' if len(df.get('statement', '')) > 200 else df.get('statement', ''),
                        'confidence': round(df.get('confidence', 0), 3),
                        'relevance_score': round(df.get('relevance_score', 0), 3),
                    }
                    for df in domain_facts
                ]
            },
            'plan': {
                'steps': plan_steps,
                'total_steps': len(plan_steps)
            },
            'metadata': {
                'created_at': timestamp,
                'embedding_dimensions': len(task_embedding),
                'stored_in_cosmos': True,
                'long_term_memory_used': len(task_instructions) > 0,
                'facts_memory_used': len(domain_facts) > 0
            }
        }
        
        return json.dumps(response, indent=2)
    
    except Exception as e:
        logger.error(f"Error in next_best_action: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def store_memory_tool(content: str, session_id: str, memory_type: str = "context") -> str:
    """
    Store information in short-term memory for later retrieval.
    
    Args:
        content: The content to remember
        session_id: The session ID to associate the memory with
        memory_type: Type of memory (context, conversation, task, plan)
    
    Returns:
        JSON response with the stored memory ID
    """
    if not short_term_memory:
        return json.dumps({"error": "Memory provider not configured"})
    
    try:
        import asyncio
        
        # Map string to MemoryType enum
        type_map = {
            "context": MemoryType.CONTEXT,
            "conversation": MemoryType.CONVERSATION,
            "task": MemoryType.TASK,
            "plan": MemoryType.PLAN,
        }
        mem_type = type_map.get(memory_type.lower(), MemoryType.CONTEXT)
        
        # Generate embedding for the content
        embedding = None
        if FOUNDRY_PROJECT_ENDPOINT:
            try:
                embedding = get_embedding(content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=mem_type,
            embedding=embedding,
            session_id=session_id,
        )
        
        # Run async store in sync context
        loop = asyncio.new_event_loop()
        entry_id = loop.run_until_complete(short_term_memory.store(entry))
        loop.close()
        
        return json.dumps({
            "success": True,
            "memory_id": entry_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "has_embedding": embedding is not None,
        })
    
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def recall_memory_tool(query: str, session_id: str, limit: int = 5) -> str:
    """
    Recall relevant memories from short-term memory based on semantic similarity.
    
    Args:
        query: The query to search for relevant memories
        session_id: The session ID to search within
        limit: Maximum number of memories to return
    
    Returns:
        JSON response with relevant memories
    """
    if not short_term_memory:
        return json.dumps({"error": "Memory provider not configured"})
    
    if not FOUNDRY_PROJECT_ENDPOINT:
        return json.dumps({"error": "Foundry endpoint not configured for embeddings"})
    
    try:
        import asyncio
        
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        
        # Search for similar memories
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(short_term_memory.search(
            query_embedding=query_embedding,
            limit=limit,
            threshold=0.6,
            session_id=session_id,
        ))
        loop.close()
        
        memories = [
            {
                "id": r.entry.id,
                "content": r.entry.content,
                "memory_type": r.entry.memory_type.value,
                "similarity_score": round(r.score, 3),
                "created_at": r.entry.created_at,
            }
            for r in results
        ]
        
        return json.dumps({
            "query": query,
            "session_id": session_id,
            "memories_found": len(memories),
            "memories": memories,
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error recalling memory: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def get_session_history_tool(session_id: str, limit: int = 20) -> str:
    """
    Get conversation history for a session.
    
    Args:
        session_id: The session ID to get history for
        limit: Maximum number of messages to return
    
    Returns:
        JSON response with conversation history
    """
    if not short_term_memory:
        return json.dumps({"error": "Memory provider not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        history = loop.run_until_complete(
            short_term_memory.get_conversation_history(session_id, limit)
        )
        loop.close()
        
        return json.dumps({
            "session_id": session_id,
            "message_count": len(history),
            "messages": history,
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def clear_session_memory_tool(session_id: str) -> str:
    """
    Clear all short-term memory for a session.
    
    Args:
        session_id: The session ID to clear
    
    Returns:
        JSON response with number of entries cleared
    """
    if not short_term_memory:
        return json.dumps({"error": "Memory provider not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        count = loop.run_until_complete(short_term_memory.clear_session(session_id))
        loop.close()
        
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "entries_cleared": count,
        })
    
    except Exception as e:
        logger.error(f"Error clearing session memory: {e}")
        return json.dumps({"error": str(e)})


# =========================================
# Fabric IQ Facts Memory Tools
# =========================================

@ai_function
def search_facts_tool(query: str, domain: str = None, limit: int = 5) -> str:
    """
    Search for relevant facts from Fabric IQ ontology-grounded knowledge.
    Uses semantic search to find facts across Customer, DevOps, and User Management domains.
    
    Args:
        query: Natural language query to search for relevant facts
        domain: Optional domain filter (customer, devops, user_management)
        limit: Maximum number of facts to return (default: 5)
    
    Returns:
        JSON response with matching facts and their relevance scores
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            facts_memory.search_facts(
                query=query,
                domain=domain,
                limit=limit,
            )
        )
        loop.close()
        
        facts_data = [
            {
                "id": r.fact.id,
                "statement": r.fact.statement,
                "domain": r.fact.domain,
                "fact_type": r.fact.fact_type,
                "confidence": round(r.fact.confidence, 3),
                "relevance_score": round(r.score, 3),
                "evidence_count": len(r.fact.evidence),
                "context": r.fact.context,
            }
            for r in results
        ]
        
        return json.dumps({
            "query": query,
            "domain_filter": domain,
            "facts_found": len(facts_data),
            "facts": facts_data,
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error searching facts: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def get_customer_churn_facts_tool(risk_level: str = None) -> str:
    """
    Retrieve customer churn analysis facts from Fabric IQ.
    Returns predictions and observations about customer churn risk.
    
    Args:
        risk_level: Optional filter by risk level (critical, high, medium, low, minimal)
    
    Returns:
        JSON response with customer churn facts and statistics
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        
        # Search for churn-related facts
        results = loop.run_until_complete(
            facts_memory.search_facts(
                query="customer churn risk prediction",
                domain="customer",
                fact_type="prediction",
                limit=20,
            )
        )
        loop.close()
        
        # Filter by risk level if specified
        facts_data = []
        for r in results:
            if risk_level:
                fact_risk = r.fact.context.get("risk_level", "")
                if fact_risk.lower() != risk_level.lower():
                    continue
            
            facts_data.append({
                "customer_id": r.fact.evidence[0] if r.fact.evidence else None,
                "statement": r.fact.statement,
                "churn_risk": r.fact.confidence,
                "risk_level": r.fact.context.get("risk_level"),
                "segment": r.fact.context.get("segment"),
                "tenure_months": r.fact.context.get("tenure_months"),
                "monthly_spend": r.fact.context.get("monthly_spend"),
            })
        
        # Calculate summary statistics
        if facts_data:
            avg_risk = sum(f["churn_risk"] for f in facts_data) / len(facts_data)
            risk_distribution = {}
            for f in facts_data:
                level = f["risk_level"]
                risk_distribution[level] = risk_distribution.get(level, 0) + 1
        else:
            avg_risk = 0
            risk_distribution = {}
        
        return json.dumps({
            "total_at_risk_customers": len(facts_data),
            "average_churn_risk": round(avg_risk, 3),
            "risk_distribution": risk_distribution,
            "filter_applied": risk_level,
            "customers": facts_data[:10],  # Limit response size
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting customer churn facts: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def get_pipeline_health_facts_tool(include_failures: bool = True) -> str:
    """
    Retrieve CI/CD pipeline health facts from Fabric IQ.
    Returns observations about pipeline success rates and failures.
    
    Args:
        include_failures: Whether to include detailed failure information (default: True)
    
    Returns:
        JSON response with pipeline health facts and failure details
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        
        # Get pipeline health observations
        health_results = loop.run_until_complete(
            facts_memory.search_facts(
                query="pipeline success rate deployment",
                domain="devops",
                fact_type="observation",
                limit=10,
            )
        )
        
        pipeline_facts = []
        for r in health_results:
            if "pipeline" in r.fact.id.lower():
                pipeline_facts.append({
                    "pipeline_id": r.fact.evidence[0] if r.fact.evidence else None,
                    "statement": r.fact.statement,
                    "success_rate": r.fact.context.get("success_rate"),
                    "total_runs": r.fact.context.get("total_runs"),
                    "failures": r.fact.context.get("failures"),
                    "service": r.fact.context.get("service"),
                    "avg_duration_seconds": r.fact.context.get("avg_duration"),
                })
        
        # Get failure details if requested
        failure_facts = []
        if include_failures:
            failure_results = loop.run_until_complete(
                facts_memory.search_facts(
                    query="pipeline failure error",
                    domain="devops",
                    fact_type="observation",
                    limit=10,
                )
            )
            
            for r in failure_results:
                if "failure" in r.fact.id.lower():
                    failure_facts.append({
                        "run_id": r.fact.evidence[0] if r.fact.evidence else None,
                        "statement": r.fact.statement,
                        "failure_category": r.fact.context.get("failure_category"),
                        "failure_stage": r.fact.context.get("failure_stage"),
                        "commit_sha": r.fact.context.get("commit_sha"),
                    })
        
        loop.close()
        
        # Calculate summary
        if pipeline_facts:
            avg_success_rate = sum(
                p["success_rate"] or 0 for p in pipeline_facts
            ) / len(pipeline_facts)
            total_failures = sum(p["failures"] or 0 for p in pipeline_facts)
        else:
            avg_success_rate = 0
            total_failures = 0
        
        return json.dumps({
            "total_pipelines": len(pipeline_facts),
            "average_success_rate": round(avg_success_rate, 3),
            "total_recent_failures": total_failures,
            "pipelines": pipeline_facts,
            "failure_details": failure_facts if include_failures else [],
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting pipeline health facts: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def get_user_security_facts_tool(include_alerts: bool = True) -> str:
    """
    Retrieve user security and access facts from Fabric IQ.
    Returns observations about user activity and security alerts.
    
    Args:
        include_alerts: Whether to include security alert facts (default: True)
    
    Returns:
        JSON response with user activity facts and security alerts
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        
        # Get user activity observations
        activity_results = loop.run_until_complete(
            facts_memory.search_facts(
                query="user login authentication activity",
                domain="user_management",
                fact_type="observation",
                limit=15,
            )
        )
        
        user_facts = []
        for r in activity_results:
            if "user-" in r.fact.id:
                user_facts.append({
                    "user_id": r.fact.evidence[0] if r.fact.evidence else None,
                    "statement": r.fact.statement,
                    "roles": r.fact.context.get("roles"),
                    "mfa_enabled": r.fact.context.get("mfa_enabled"),
                    "status": r.fact.context.get("status"),
                    "login_successes": r.fact.context.get("login_successes"),
                    "login_failures": r.fact.context.get("login_failures"),
                    "high_risk_events": r.fact.context.get("high_risk_events"),
                })
        
        # Get security alerts if requested
        alert_facts = []
        if include_alerts:
            alert_results = loop.run_until_complete(
                facts_memory.search_facts(
                    query="security alert suspicious activity",
                    domain="user_management",
                    fact_type="derived",
                    limit=10,
                )
            )
            
            for r in alert_results:
                if "security" in r.fact.id.lower():
                    alert_facts.append({
                        "user_id": r.fact.evidence[0] if r.fact.evidence else None,
                        "statement": r.fact.statement,
                        "alert_type": r.fact.context.get("alert_type"),
                        "failed_logins": r.fact.context.get("failed_logins"),
                        "high_risk_events": r.fact.context.get("high_risk_events"),
                        "confidence": r.fact.confidence,
                    })
        
        loop.close()
        
        # Calculate summary
        total_users = len(user_facts)
        mfa_enabled_count = sum(1 for u in user_facts if u.get("mfa_enabled"))
        high_risk_users = len(alert_facts)
        
        return json.dumps({
            "total_users_analyzed": total_users,
            "mfa_adoption_rate": round(mfa_enabled_count / total_users, 3) if total_users else 0,
            "security_alerts_count": high_risk_users,
            "users": user_facts[:10],  # Limit response size
            "security_alerts": alert_facts if include_alerts else [],
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting user security facts: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def cross_domain_analysis_tool(query: str, source_domain: str, target_domain: str) -> str:
    """
    Perform cross-domain reasoning to find connections between different domains.
    Leverages Fabric IQ's graph capabilities for entity-relationship traversal.
    
    Args:
        query: Natural language query describing the connection to find
        source_domain: Starting domain (customer, devops, user_management)
        target_domain: Target domain to connect to
    
    Returns:
        JSON response with cross-domain connections and insights
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        import asyncio
        
        loop = asyncio.new_event_loop()
        connections = loop.run_until_complete(
            facts_memory.cross_domain_query(
                query=query,
                source_domain=source_domain,
                target_domain=target_domain,
            )
        )
        loop.close()
        
        return json.dumps({
            "query": query,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "connections_found": len(connections),
            "connections": connections[:5],  # Limit to top 5 connections
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error in cross-domain analysis: {e}")
        return json.dumps({"error": str(e)})


@ai_function
def get_facts_memory_stats_tool() -> str:
    """
    Get statistics about the Fabric IQ Facts Memory.
    Returns counts of entities and facts by domain.
    
    Returns:
        JSON response with facts memory statistics
    """
    if not facts_memory:
        return json.dumps({"error": "Facts memory not configured"})
    
    try:
        stats = facts_memory.get_stats()
        
        return json.dumps({
            "total_entities": stats["total_entities"],
            "total_relationships": stats["total_relationships"],
            "total_facts": stats["total_facts"],
            "entities_by_type": stats["entities_by_type"],
            "entities_by_domain": stats["entities_by_domain"],
            "facts_by_domain": stats["facts_by_domain"],
            "ontology_name": FABRIC_ONTOLOGY_NAME,
            "fabric_endpoint_configured": bool(FABRIC_ENDPOINT),
        }, indent=2)
    
    except Exception as e:
        logger.error(f"Error getting facts memory stats: {e}")
        return json.dumps({"error": str(e)})


# Create the AI Agent with tools
def create_mcp_agent():
    """Create and configure the MCP AI Agent with Microsoft Agent Framework."""
    if not FOUNDRY_PROJECT_ENDPOINT:
        logger.warning("FOUNDRY_PROJECT_ENDPOINT not configured - AI Agent will not be available")
        return None
    
    try:
        agent_credential = DefaultAzureCredential()
        client = AzureAIAgentClient(
            endpoint=FOUNDRY_PROJECT_ENDPOINT,
            credential=agent_credential,
        )
        logger.info("MCP AI Agent Client created successfully")
        return client
    except Exception as e:
        logger.error(f"Error creating AI Agent: {e}")
        return None


# Initialize the AI agent client (will be set on startup)
mcp_ai_agent = None


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


# Define MCP tools
TOOLS = [
    MCPTool(
        name="hello_mcp",
        description="Hello world MCP tool.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    MCPTool(
        name="get_snippet",
        description="Retrieve a snippet by name from Azure Blob Storage.",
        inputSchema={
            "type": "object",
            "properties": {
                "snippetname": {
                    "type": "string",
                    "description": "The name of the snippet to retrieve"
                }
            },
            "required": ["snippetname"]
        }
    ),
    MCPTool(
        name="save_snippet",
        description="Save a snippet with a name to Azure Blob Storage.",
        inputSchema={
            "type": "object",
            "properties": {
                "snippetname": {
                    "type": "string",
                    "description": "The name of the snippet"
                },
                "snippet": {
                    "type": "string",
                    "description": "The content of the snippet"
                }
            },
            "required": ["snippetname", "snippet"]
        }
    ),
    MCPTool(
        name="ask_foundry",
        description="Ask a question and get an answer using the Azure AI Foundry model.",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the AI model"
                }
            },
            "required": ["question"]
        }
    ),
    MCPTool(
        name="next_best_action",
        description="Analyze a task using semantic reasoning with embeddings and ontology-grounded facts. Uses three memory layers: (1) Short-term memory - finds similar past tasks from CosmosDB, (2) Long-term memory - retrieves task instructions from AI Search, (3) Facts memory - queries domain facts from Fabric IQ ontologies (Customer Churn, CI/CD Pipelines, User Management). Generates a comprehensive plan grounded in domain knowledge. Returns task analysis, similar tasks, domain facts, and planned steps.",
        inputSchema={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task description in natural language (English sentence) to analyze and plan"
                }
            },
            "required": ["task"]
        }
    ),
    MCPTool(
        name="store_memory",
        description="Store information in short-term memory for later retrieval. Useful for remembering context, user preferences, or intermediate results within a session.",
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                },
                "session_id": {
                    "type": "string",
                    "description": "The session ID to associate the memory with"
                },
                "memory_type": {
                    "type": "string",
                    "description": "Type of memory: context, conversation, task, or plan",
                    "enum": ["context", "conversation", "task", "plan"]
                }
            },
            "required": ["content", "session_id"]
        }
    ),
    MCPTool(
        name="recall_memory",
        description="Recall relevant memories from short-term memory based on semantic similarity. Returns memories that are contextually related to the query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for relevant memories"
                },
                "session_id": {
                    "type": "string",
                    "description": "The session ID to search within"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default: 5)"
                }
            },
            "required": ["query", "session_id"]
        }
    ),
    MCPTool(
        name="get_session_history",
        description="Get conversation history for a session. Returns the messages exchanged in the session.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to get history for"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of messages to return (default: 20)"
                }
            },
            "required": ["session_id"]
        }
    ),
    MCPTool(
        name="clear_session_memory",
        description="Clear all short-term memory for a session. Use when starting fresh or cleaning up.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The session ID to clear"
                }
            },
            "required": ["session_id"]
        }
    ),
    # =========================================
    # Fabric IQ Facts Memory Tools
    # =========================================
    MCPTool(
        name="search_facts",
        description="Search for relevant facts from Fabric IQ ontology-grounded knowledge. Uses semantic search to find facts across Customer, DevOps, and User Management domains.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query to search for relevant facts"
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain filter (customer, devops, user_management)",
                    "enum": ["customer", "devops", "user_management"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of facts to return (default: 5)"
                }
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="get_customer_churn_facts",
        description="Retrieve customer churn analysis facts from Fabric IQ. Returns predictions and observations about customer churn risk with segment analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "risk_level": {
                    "type": "string",
                    "description": "Optional filter by risk level",
                    "enum": ["critical", "high", "medium", "low", "minimal"]
                }
            },
            "required": []
        }
    ),
    MCPTool(
        name="get_pipeline_health_facts",
        description="Retrieve CI/CD pipeline health facts from Fabric IQ. Returns observations about pipeline success rates, failures, and deployment status.",
        inputSchema={
            "type": "object",
            "properties": {
                "include_failures": {
                    "type": "boolean",
                    "description": "Whether to include detailed failure information (default: true)"
                }
            },
            "required": []
        }
    ),
    MCPTool(
        name="get_user_security_facts",
        description="Retrieve user security and access facts from Fabric IQ. Returns observations about user activity, authentication patterns, and security alerts.",
        inputSchema={
            "type": "object",
            "properties": {
                "include_alerts": {
                    "type": "boolean",
                    "description": "Whether to include security alert facts (default: true)"
                }
            },
            "required": []
        }
    ),
    MCPTool(
        name="cross_domain_analysis",
        description="Perform cross-domain reasoning to find connections between different domains. Leverages Fabric IQ's graph capabilities for entity-relationship traversal.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing the connection to find"
                },
                "source_domain": {
                    "type": "string",
                    "description": "Starting domain for analysis",
                    "enum": ["customer", "devops", "user_management"]
                },
                "target_domain": {
                    "type": "string",
                    "description": "Target domain to connect to",
                    "enum": ["customer", "devops", "user_management"]
                }
            },
            "required": ["query", "source_domain", "target_domain"]
        }
    ),
    MCPTool(
        name="get_facts_memory_stats",
        description="Get statistics about the Fabric IQ Facts Memory. Returns counts of entities and facts by domain.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
]


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    """Execute an MCP tool"""
    try:
        if tool_name == "hello_mcp":
            return MCPToolResult(
                content=[{
                    "type": "text",
                    "text": "Hello I am MCPTool!"
                }]
            )
        
        elif tool_name == "get_snippet":
            snippet_name = arguments.get("snippetname")
            if not snippet_name:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No snippet name provided"}],
                    isError=True
                )
            
            if not blob_service_client:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Storage not configured"}],
                    isError=True
                )
            
            try:
                blob_client = blob_service_client.get_blob_client(
                    container=SNIPPETS_CONTAINER,
                    blob=f"{snippet_name}.json"
                )
                blob_data = blob_client.download_blob().readall()
                snippet_content = blob_data.decode('utf-8')
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": snippet_content
                    }]
                )
            except Exception as e:
                logger.error(f"Error retrieving snippet: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error retrieving snippet: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "save_snippet":
            snippet_name = arguments.get("snippetname")
            snippet_content = arguments.get("snippet")
            
            if not snippet_name:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No snippet name provided"}],
                    isError=True
                )
            
            if not snippet_content:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No snippet content provided"}],
                    isError=True
                )
            
            if not blob_service_client:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Storage not configured"}],
                    isError=True
                )
            
            try:
                blob_client = blob_service_client.get_blob_client(
                    container=SNIPPETS_CONTAINER,
                    blob=f"{snippet_name}.json"
                )
                blob_client.upload_blob(snippet_content.encode('utf-8'), overwrite=True)
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": f"Snippet '{snippet_name}' saved successfully"
                    }]
                )
            except Exception as e:
                logger.error(f"Error saving snippet: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error saving snippet: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "ask_foundry":
            question = arguments.get("question")
            if not question:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No question provided"}],
                    isError=True
                )
            
            if not FOUNDRY_PROJECT_ENDPOINT:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Foundry endpoint not configured"}],
                    isError=True
                )
            
            try:
                from openai import AzureOpenAI
                
                credential = DefaultAzureCredential()
                # Get a token for Azure Cognitive Services
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                
                # Extract the base endpoint (remove /api/projects/proj-default if present)
                # Use the services.ai.azure.com endpoint directly
                base_endpoint = FOUNDRY_PROJECT_ENDPOINT.split('/api/projects')[0] if '/api/projects' in FOUNDRY_PROJECT_ENDPOINT else FOUNDRY_PROJECT_ENDPOINT
                
                logger.info(f"Using Foundry endpoint: {base_endpoint}")
                
                client = AzureOpenAI(
                    azure_endpoint=base_endpoint,
                    api_key=token.token,
                    api_version="2024-02-15-preview"
                )
                
                response = client.chat.completions.create(
                    model=FOUNDRY_MODEL_DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": question}]
                )
                
                answer = "No response generated"
                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": answer
                    }]
                )
            except Exception as e:
                logger.error(f"Error calling Foundry model: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error calling Foundry model: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "next_best_action":
            task = arguments.get("task")
            if not task:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No task provided"}],
                    isError=True
                )
            
            if not FOUNDRY_PROJECT_ENDPOINT:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Foundry endpoint not configured"}],
                    isError=True
                )
            
            if not cosmos_tasks_container or not cosmos_plans_container:
                return MCPToolResult(
                    content=[{"type": "text", "text": "CosmosDB not configured"}],
                    isError=True
                )
            
            try:
                task_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                
                # Step 1: Generate embedding for the task
                logger.info(f"Generating embedding for task: {task[:100]}...")
                task_embedding = get_embedding(task)
                
                # Step 2: Analyze intent
                logger.info("Analyzing task intent...")
                intent = analyze_intent(task)
                
                # Step 3: Find similar tasks using cosine similarity
                logger.info("Searching for similar past tasks...")
                similar_tasks = find_similar_tasks(task_embedding, threshold=0.7, limit=5)
                
                # Step 4: Generate plan based on task and similar past tasks
                logger.info("Generating execution plan...")
                plan_steps = generate_plan(task, similar_tasks)
                
                # Step 5: Store task in CosmosDB
                task_doc = {
                    'id': task_id,
                    'task': task,
                    'intent': intent,
                    'embedding': task_embedding,
                    'created_at': timestamp,
                    'similar_task_count': len(similar_tasks)
                }
                cosmos_tasks_container.upsert_item(task_doc)
                logger.info(f"Task stored in CosmosDB with id: {task_id}")
                
                # Step 6: Store plan in CosmosDB
                plan_doc = {
                    'id': str(uuid.uuid4()),
                    'taskId': task_id,
                    'task': task,
                    'intent': intent,
                    'steps': plan_steps,
                    'similar_tasks_referenced': [{'id': st['id'], 'similarity': st['similarity']} for st in similar_tasks],
                    'created_at': timestamp,
                    'status': 'planned'
                }
                cosmos_plans_container.upsert_item(plan_doc)
                logger.info(f"Plan stored in CosmosDB for task: {task_id}")
                
                # Build response
                response = {
                    'task_id': task_id,
                    'task': task,
                    'intent': intent,
                    'analysis': {
                        'similar_tasks_found': len(similar_tasks),
                        'similar_tasks': [
                            {
                                'task': st['task'],
                                'intent': st['intent'],
                                'similarity_score': round(st['similarity'], 3)
                            }
                            for st in similar_tasks
                        ]
                    },
                    'plan': {
                        'steps': plan_steps,
                        'total_steps': len(plan_steps)
                    },
                    'metadata': {
                        'created_at': timestamp,
                        'embedding_dimensions': len(task_embedding),
                        'stored_in_cosmos': True
                    }
                }
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps(response, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error in next_best_action: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error in next_best_action: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "store_memory":
            content = arguments.get("content")
            session_id = arguments.get("session_id")
            memory_type = arguments.get("memory_type", "context")
            
            if not content:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No content provided"}],
                    isError=True
                )
            
            if not session_id:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No session_id provided"}],
                    isError=True
                )
            
            if not short_term_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Memory provider not configured"}],
                    isError=True
                )
            
            try:
                # Map string to MemoryType enum
                type_map = {
                    "context": MemoryType.CONTEXT,
                    "conversation": MemoryType.CONVERSATION,
                    "task": MemoryType.TASK,
                    "plan": MemoryType.PLAN,
                }
                mem_type = type_map.get(memory_type.lower(), MemoryType.CONTEXT)
                
                # Generate embedding for the content
                embedding = None
                if FOUNDRY_PROJECT_ENDPOINT:
                    try:
                        embedding = get_embedding(content)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")
                
                entry = MemoryEntry(
                    id=str(uuid.uuid4()),
                    content=content,
                    memory_type=mem_type,
                    embedding=embedding,
                    session_id=session_id,
                )
                
                entry_id = await short_term_memory.store(entry)
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "memory_id": entry_id,
                            "session_id": session_id,
                            "memory_type": memory_type,
                            "has_embedding": embedding is not None,
                        })
                    }]
                )
            except Exception as e:
                logger.error(f"Error storing memory: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error storing memory: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "recall_memory":
            query = arguments.get("query")
            session_id = arguments.get("session_id")
            limit = arguments.get("limit", 5)
            
            if not query:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No query provided"}],
                    isError=True
                )
            
            if not session_id:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No session_id provided"}],
                    isError=True
                )
            
            if not short_term_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Memory provider not configured"}],
                    isError=True
                )
            
            if not FOUNDRY_PROJECT_ENDPOINT:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Foundry endpoint not configured for embeddings"}],
                    isError=True
                )
            
            try:
                query_embedding = get_embedding(query)
                
                results = await short_term_memory.search(
                    query_embedding=query_embedding,
                    limit=limit,
                    threshold=0.6,
                    session_id=session_id,
                )
                
                memories = [
                    {
                        "id": r.entry.id,
                        "content": r.entry.content,
                        "memory_type": r.entry.memory_type.value,
                        "similarity_score": round(r.score, 3),
                        "created_at": r.entry.created_at,
                    }
                    for r in results
                ]
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "query": query,
                            "session_id": session_id,
                            "memories_found": len(memories),
                            "memories": memories,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error recalling memory: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error recalling memory: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "get_session_history":
            session_id = arguments.get("session_id")
            limit = arguments.get("limit", 20)
            
            if not session_id:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No session_id provided"}],
                    isError=True
                )
            
            if not short_term_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Memory provider not configured"}],
                    isError=True
                )
            
            try:
                history = await short_term_memory.get_conversation_history(session_id, limit)
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "session_id": session_id,
                            "message_count": len(history),
                            "messages": history,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error getting session history: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error getting session history: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "clear_session_memory":
            session_id = arguments.get("session_id")
            
            if not session_id:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No session_id provided"}],
                    isError=True
                )
            
            if not short_term_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Memory provider not configured"}],
                    isError=True
                )
            
            try:
                count = await short_term_memory.clear_session(session_id)
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "success": True,
                            "session_id": session_id,
                            "entries_cleared": count,
                        })
                    }]
                )
            except Exception as e:
                logger.error(f"Error clearing session memory: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error clearing session memory: {str(e)}"}],
                    isError=True
                )
        
        # =========================================
        # Fabric IQ Facts Memory Tool Handlers
        # =========================================
        
        elif tool_name == "search_facts":
            query = arguments.get("query")
            domain = arguments.get("domain")
            limit = arguments.get("limit", 5)
            
            if not query:
                return MCPToolResult(
                    content=[{"type": "text", "text": "No query provided"}],
                    isError=True
                )
            
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                results = await facts_memory.search_facts(
                    query=query,
                    domain=domain,
                    limit=limit,
                )
                
                facts_data = [
                    {
                        "id": r.fact.id,
                        "statement": r.fact.statement,
                        "domain": r.fact.domain,
                        "fact_type": r.fact.fact_type,
                        "confidence": round(r.fact.confidence, 3),
                        "relevance_score": round(r.score, 3),
                        "context": r.fact.context,
                    }
                    for r in results
                ]
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "query": query,
                            "domain_filter": domain,
                            "facts_found": len(facts_data),
                            "facts": facts_data,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error searching facts: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error searching facts: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "get_customer_churn_facts":
            risk_level = arguments.get("risk_level")
            
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                results = await facts_memory.search_facts(
                    query="customer churn risk prediction",
                    domain="customer",
                    fact_type="prediction",
                    limit=20,
                )
                
                facts_data = []
                for r in results:
                    if risk_level:
                        fact_risk = r.fact.context.get("risk_level", "")
                        if fact_risk.lower() != risk_level.lower():
                            continue
                    
                    facts_data.append({
                        "customer_id": r.fact.evidence[0] if r.fact.evidence else None,
                        "statement": r.fact.statement,
                        "churn_risk": r.fact.confidence,
                        "risk_level": r.fact.context.get("risk_level"),
                        "segment": r.fact.context.get("segment"),
                        "tenure_months": r.fact.context.get("tenure_months"),
                        "monthly_spend": r.fact.context.get("monthly_spend"),
                    })
                
                avg_risk = sum(f["churn_risk"] for f in facts_data) / len(facts_data) if facts_data else 0
                risk_distribution = {}
                for f in facts_data:
                    level = f["risk_level"]
                    risk_distribution[level] = risk_distribution.get(level, 0) + 1
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "total_at_risk_customers": len(facts_data),
                            "average_churn_risk": round(avg_risk, 3),
                            "risk_distribution": risk_distribution,
                            "filter_applied": risk_level,
                            "customers": facts_data[:10],
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error getting customer churn facts: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error getting customer churn facts: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "get_pipeline_health_facts":
            include_failures = arguments.get("include_failures", True)
            
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                # Get pipeline health observations
                health_results = await facts_memory.search_facts(
                    query="pipeline success rate deployment",
                    domain="devops",
                    fact_type="observation",
                    limit=10,
                )
                
                pipeline_facts = []
                for r in health_results:
                    if "pipeline" in r.fact.id.lower():
                        pipeline_facts.append({
                            "pipeline_id": r.fact.evidence[0] if r.fact.evidence else None,
                            "statement": r.fact.statement,
                            "success_rate": r.fact.context.get("success_rate"),
                            "total_runs": r.fact.context.get("total_runs"),
                            "failures": r.fact.context.get("failures"),
                            "service": r.fact.context.get("service"),
                        })
                
                failure_facts = []
                if include_failures:
                    failure_results = await facts_memory.search_facts(
                        query="pipeline failure error",
                        domain="devops",
                        fact_type="observation",
                        limit=10,
                    )
                    
                    for r in failure_results:
                        if "failure" in r.fact.id.lower():
                            failure_facts.append({
                                "run_id": r.fact.evidence[0] if r.fact.evidence else None,
                                "statement": r.fact.statement,
                                "failure_category": r.fact.context.get("failure_category"),
                                "failure_stage": r.fact.context.get("failure_stage"),
                            })
                
                avg_success = sum(p["success_rate"] or 0 for p in pipeline_facts) / len(pipeline_facts) if pipeline_facts else 0
                total_failures = sum(p["failures"] or 0 for p in pipeline_facts)
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "total_pipelines": len(pipeline_facts),
                            "average_success_rate": round(avg_success, 3),
                            "total_recent_failures": total_failures,
                            "pipelines": pipeline_facts,
                            "failure_details": failure_facts,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error getting pipeline health facts: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error getting pipeline health facts: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "get_user_security_facts":
            include_alerts = arguments.get("include_alerts", True)
            
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                activity_results = await facts_memory.search_facts(
                    query="user login authentication activity",
                    domain="user_management",
                    fact_type="observation",
                    limit=15,
                )
                
                user_facts = []
                for r in activity_results:
                    if "user-" in r.fact.id:
                        user_facts.append({
                            "user_id": r.fact.evidence[0] if r.fact.evidence else None,
                            "statement": r.fact.statement,
                            "roles": r.fact.context.get("roles"),
                            "mfa_enabled": r.fact.context.get("mfa_enabled"),
                            "status": r.fact.context.get("status"),
                            "login_successes": r.fact.context.get("login_successes"),
                            "login_failures": r.fact.context.get("login_failures"),
                        })
                
                alert_facts = []
                if include_alerts:
                    alert_results = await facts_memory.search_facts(
                        query="security alert suspicious activity",
                        domain="user_management",
                        fact_type="derived",
                        limit=10,
                    )
                    
                    for r in alert_results:
                        if "security" in r.fact.id.lower():
                            alert_facts.append({
                                "user_id": r.fact.evidence[0] if r.fact.evidence else None,
                                "statement": r.fact.statement,
                                "alert_type": r.fact.context.get("alert_type"),
                                "confidence": r.fact.confidence,
                            })
                
                total_users = len(user_facts)
                mfa_count = sum(1 for u in user_facts if u.get("mfa_enabled"))
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "total_users_analyzed": total_users,
                            "mfa_adoption_rate": round(mfa_count / total_users, 3) if total_users else 0,
                            "security_alerts_count": len(alert_facts),
                            "users": user_facts[:10],
                            "security_alerts": alert_facts,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error getting user security facts: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error getting user security facts: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "cross_domain_analysis":
            query = arguments.get("query")
            source_domain = arguments.get("source_domain")
            target_domain = arguments.get("target_domain")
            
            if not all([query, source_domain, target_domain]):
                return MCPToolResult(
                    content=[{"type": "text", "text": "Missing required parameters: query, source_domain, target_domain"}],
                    isError=True
                )
            
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                connections = await facts_memory.cross_domain_query(
                    query=query,
                    source_domain=source_domain,
                    target_domain=target_domain,
                )
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "query": query,
                            "source_domain": source_domain,
                            "target_domain": target_domain,
                            "connections_found": len(connections),
                            "connections": connections[:5],
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error in cross-domain analysis: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error in cross-domain analysis: {str(e)}"}],
                    isError=True
                )
        
        elif tool_name == "get_facts_memory_stats":
            if not facts_memory:
                return MCPToolResult(
                    content=[{"type": "text", "text": "Facts memory not configured"}],
                    isError=True
                )
            
            try:
                stats = facts_memory.get_stats()
                
                return MCPToolResult(
                    content=[{
                        "type": "text",
                        "text": json.dumps({
                            "total_entities": stats["total_entities"],
                            "total_relationships": stats["total_relationships"],
                            "total_facts": stats["total_facts"],
                            "entities_by_type": stats["entities_by_type"],
                            "entities_by_domain": stats["entities_by_domain"],
                            "facts_by_domain": stats["facts_by_domain"],
                            "ontology_name": FABRIC_ONTOLOGY_NAME,
                        }, indent=2)
                    }]
                )
            except Exception as e:
                logger.error(f"Error getting facts memory stats: {e}")
                return MCPToolResult(
                    content=[{"type": "text", "text": f"Error getting facts memory stats: {str(e)}"}],
                    isError=True
                )
        
        else:
            return MCPToolResult(
                content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                isError=True
            )
    
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return MCPToolResult(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            isError=True
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/runtime/webhooks/mcp/sse")
async def mcp_sse_endpoint(request: Request):
    """
    SSE endpoint for MCP protocol
    Establishes a long-lived connection for server-sent events
    """
    session_id = str(uuid.uuid4())
    logger.info(f"New SSE session established: {session_id}")
    
    # Store session
    sessions[session_id] = {
        "created_at": datetime.utcnow().isoformat(),
        "message_queue": asyncio.Queue()
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
                    # Wait for messages with timeout
                    message = await asyncio.wait_for(
                        sessions[session_id]["message_queue"].get(),
                        timeout=30.0
                    )
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    
        except asyncio.CancelledError:
            logger.info(f"SSE connection cancelled for session {session_id}")
        finally:
            # Cleanup session
            if session_id in sessions:
                del sessions[session_id]
            logger.info(f"SSE session closed: {session_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/runtime/webhooks/mcp/message")
async def mcp_message_endpoint(request: Request):
    """
    Message endpoint for MCP protocol
    Handles JSON-RPC 2.0 requests
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
                    "id": request_id
                }
            )
        
        # Handle initialize
        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mcp-server",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }
            return JSONResponse(content=response)
        
        # Handle tools/list
        elif method == "tools/list":
            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
                for tool in TOOLS
            ]
            
            response = {
                "jsonrpc": "2.0",
                "result": {
                    "tools": tools_list
                },
                "id": request_id
            }
            return JSONResponse(content=response)
        
        # Handle tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            # Execute the tool
            result = await execute_tool(tool_name, arguments)
            
            response = {
                "jsonrpc": "2.0",
                "result": asdict(result),
                "id": request_id
            }
            return JSONResponse(content=response)
        
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id
                }
            )
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                "id": body.get("id") if 'body' in locals() else None
            }
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "MCP Server",
        "version": "1.0.0",
        "endpoints": {
            "sse": "/runtime/webhooks/mcp/sse",
            "message": "/runtime/webhooks/mcp/message",
            "health": "/health",
            "agent_chat": "/agent/chat"
        },
        "agent_enabled": mcp_ai_agent is not None
    }


@app.on_event("startup")
async def startup_event():
    """Initialize the AI agent and memory providers on startup."""
    global mcp_ai_agent
    
    # Initialize AI Agent
    mcp_ai_agent = create_mcp_agent()
    if mcp_ai_agent:
        logger.info("AI Agent initialized successfully on startup")
    else:
        logger.warning("AI Agent not initialized - check FOUNDRY_PROJECT_ENDPOINT configuration")
    
    # Configure embedding function for memory provider
    if short_term_memory and FOUNDRY_PROJECT_ENDPOINT:
        short_term_memory.set_embedding_function(get_embedding)
        logger.info("Memory provider embedding function configured")
    
    # Log memory provider status
    if composite_memory:
        health = await composite_memory.health_check()
        for provider, is_healthy in health.items():
            status = "healthy" if is_healthy else "unhealthy"
            logger.info(f"Memory provider '{provider}': {status}")


@app.post("/agent/chat")
async def agent_chat(request: Request):
    """
    Chat endpoint for Microsoft Agent Framework.
    Processes user messages using the AI agent with tool capabilities.
    """
    if mcp_ai_agent is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "AI Agent not available",
                "message": "Configure FOUNDRY_PROJECT_ENDPOINT and install agent-framework packages to enable AI Agent"
            }
        )
    
    try:
        body = await request.json()
        user_message = body.get("message", "")
        conversation_history = body.get("history", [])
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "No message provided"}
            )
        
        # Build messages list for the agent
        messages = []
        
        # Add conversation history
        for hist_msg in conversation_history:
            messages.append({
                "role": hist_msg.get("role", "user"),
                "content": hist_msg.get("content", "")
            })
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Run the agent
        response = await mcp_ai_agent.run(messages)
        
        # Extract assistant response
        assistant_responses = []
        if hasattr(response, 'messages'):
            for msg in response.messages:
                if hasattr(msg, 'role') and str(msg.role).lower() == 'assistant':
                    if hasattr(msg, 'contents'):
                        for content in msg.contents:
                            if hasattr(content, 'text'):
                                assistant_responses.append(content.text)
                    elif hasattr(msg, 'content'):
                        assistant_responses.append(str(msg.content))
        
        return JSONResponse(content={
            "response": "\n".join(assistant_responses) if assistant_responses else "No response generated",
            "message_id": str(uuid.uuid4())
        })
        
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Agent error: {str(e)}"}
        )


@app.post("/agent/chat/stream")
async def agent_chat_stream(request: Request):
    """
    Streaming chat endpoint for Microsoft Agent Framework.
    Returns responses as Server-Sent Events for real-time streaming.
    """
    if mcp_ai_agent is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "AI Agent not available",
                "message": "Configure FOUNDRY_PROJECT_ENDPOINT and install agent-framework packages to enable AI Agent"
            }
        )
    
    try:
        body = await request.json()
        user_message = body.get("message", "")
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "No message provided"}
            )
        
        messages = [{"role": "user", "content": user_message}]
        
        async def generate_stream():
            try:
                async for event in mcp_ai_agent.run_stream(messages):
                    if hasattr(event, 'data') and hasattr(event.data, 'contents'):
                        for content in event.data.contents:
                            if hasattr(content, 'text'):
                                yield f"data: {json.dumps({'text': content.text})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in agent chat stream: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Agent error: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
