"""
Microsoft Fabric Data Agents - MCP Tools

Implements MCP tools for Fabric Data Agents to interact with Microsoft Fabric's data platform:
- Lakehouse Agent: Query/write Spark SQL against Fabric Lakehouses
- Warehouse Agent: Execute T-SQL queries against Fabric Data Warehouses
- Pipeline Agent: Trigger, monitor, and manage Fabric Data Pipelines
- Semantic Model Agent: Query Power BI semantic models via DAX/MDX

Security:
- Uses workload identity (DefaultAzureCredential) for authentication
- All operations are audited through Azure Monitor
- Least-privilege RBAC enforced at infrastructure level
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum

from azure.identity import DefaultAzureCredential
import requests

logger = logging.getLogger(__name__)

# =========================================
# Configuration
# =========================================

FABRIC_ENABLED = os.getenv("FABRIC_ENABLED", "false").lower() == "true"
FABRIC_DATA_AGENTS_ENABLED = os.getenv("FABRIC_DATA_AGENTS_ENABLED", "false").lower() == "true"
FABRIC_API_ENDPOINT = os.getenv("FABRIC_API_ENDPOINT", "https://api.fabric.microsoft.com/v1")
FABRIC_WORKSPACE_ID = os.getenv("FABRIC_WORKSPACE_ID", "")
FABRIC_ONELAKE_DFS_ENDPOINT = os.getenv("FABRIC_ONELAKE_DFS_ENDPOINT", "https://onelake.dfs.fabric.microsoft.com")

# Fabric resource scope for authentication
FABRIC_RESOURCE_SCOPE = "https://analysis.windows.net/powerbi/api/.default"

# =========================================
# Fabric Agent Types
# =========================================

class FabricAgentType(Enum):
    """Types of Fabric Data Agents"""
    LAKEHOUSE = "lakehouse"
    WAREHOUSE = "warehouse"
    PIPELINE = "pipeline"
    SEMANTIC_MODEL = "semantic_model"


class PipelineRunStatus(Enum):
    """Status values for Fabric pipeline runs"""
    NOT_STARTED = "NotStarted"
    IN_PROGRESS = "InProgress"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


# =========================================
# Fabric API Client
# =========================================

class FabricAPIClient:
    """Client for Microsoft Fabric REST API"""
    
    def __init__(self):
        """Initialize Fabric API client with workload identity"""
        self.credential = DefaultAzureCredential()
        self.api_endpoint = FABRIC_API_ENDPOINT
        self.workspace_id = FABRIC_WORKSPACE_ID
        self._token_cache = None
        self._token_expiry = None
    
    def _get_token(self) -> str:
        """
        Get access token for Fabric API with caching.
        
        TODO: Implement thread-safe token cache with proper locking for concurrent requests
        and integrate with azure-identity's built-in token caching mechanisms.
        """
        # Check if cached token is still valid
        if self._token_cache and self._token_expiry:
            if datetime.now(timezone.utc) < self._token_expiry:
                return self._token_cache
        
        token = self.credential.get_token(FABRIC_RESOURCE_SCOPE)
        self._token_cache = token.token
        # token.expires_on is an absolute timestamp (seconds since epoch)
        # Refresh 5 minutes before expiry
        self._token_expiry = datetime.fromtimestamp(token.expires_on, tz=timezone.utc) - timedelta(minutes=5)
        return token.token
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to Fabric API"""
        token = self._get_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.api_endpoint}/{endpoint}"
        
        logger.info(f"Fabric API {method} request to {url}")
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params,
            timeout=30
        )
        
        response.raise_for_status()
        
        if response.content:
            return response.json()
        return {}
    
    def query_lakehouse(
        self,
        lakehouse_id: str,
        query: str,
        lakehouse_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Spark SQL query against a Fabric Lakehouse.
        
        Args:
            lakehouse_id: ID of the lakehouse
            query: Spark SQL query to execute
            lakehouse_name: Optional name for logging
        
        Returns:
            Query results with schema and data
        """
        if not FABRIC_DATA_AGENTS_ENABLED:
            raise ValueError("Fabric Data Agents are not enabled")
        
        logger.info(f"Querying lakehouse {lakehouse_name or lakehouse_id}: {query[:100]}...")
        
        endpoint = f"workspaces/{self.workspace_id}/lakehouses/{lakehouse_id}/query"
        payload = {
            "query": query,
            "language": "SparkSQL"
        }
        
        result = self._make_request("POST", endpoint, data=payload)
        
        logger.info(f"Lakehouse query completed successfully")
        return result
    
    def query_warehouse(
        self,
        warehouse_id: str,
        query: str,
        warehouse_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute T-SQL query against a Fabric Data Warehouse.
        
        Args:
            warehouse_id: ID of the warehouse
            query: T-SQL query to execute
            warehouse_name: Optional name for logging
        
        Returns:
            Query results with schema and data
        """
        if not FABRIC_DATA_AGENTS_ENABLED:
            raise ValueError("Fabric Data Agents are not enabled")
        
        logger.info(f"Querying warehouse {warehouse_name or warehouse_id}: {query[:100]}...")
        
        endpoint = f"workspaces/{self.workspace_id}/warehouses/{warehouse_id}/query"
        payload = {
            "query": query,
            "language": "T-SQL"
        }
        
        result = self._make_request("POST", endpoint, data=payload)
        
        logger.info(f"Warehouse query completed successfully")
        return result
    
    def trigger_pipeline(
        self,
        pipeline_id: str,
        pipeline_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger a Fabric Data Pipeline execution.
        
        Args:
            pipeline_id: ID of the pipeline
            pipeline_name: Optional name for logging
            parameters: Optional parameters to pass to the pipeline
        
        Returns:
            Pipeline run information including run ID
        """
        if not FABRIC_DATA_AGENTS_ENABLED:
            raise ValueError("Fabric Data Agents are not enabled")
        
        logger.info(f"Triggering pipeline {pipeline_name or pipeline_id}")
        
        endpoint = f"workspaces/{self.workspace_id}/datapipelines/{pipeline_id}/run"
        payload = {
            "parameters": parameters or {}
        }
        
        result = self._make_request("POST", endpoint, data=payload)
        
        run_id = result.get("runId", "unknown")
        logger.info(f"Pipeline triggered successfully. Run ID: {run_id}")
        
        return result
    
    def get_pipeline_status(
        self,
        pipeline_id: str,
        run_id: str,
        pipeline_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get status of a Fabric Data Pipeline run.
        
        Args:
            pipeline_id: ID of the pipeline
            run_id: ID of the pipeline run
            pipeline_name: Optional name for logging
        
        Returns:
            Pipeline run status and details
        """
        if not FABRIC_DATA_AGENTS_ENABLED:
            raise ValueError("Fabric Data Agents are not enabled")
        
        logger.info(f"Getting status for pipeline {pipeline_name or pipeline_id}, run {run_id}")
        
        endpoint = f"workspaces/{self.workspace_id}/datapipelines/{pipeline_id}/runs/{run_id}"
        
        result = self._make_request("GET", endpoint)
        
        status = result.get("status", "Unknown")
        logger.info(f"Pipeline run status: {status}")
        
        return result
    
    def query_semantic_model(
        self,
        dataset_id: str,
        query: str,
        query_language: str = "DAX",
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query a Power BI semantic model (dataset) using DAX or MDX.
        
        Args:
            dataset_id: ID of the semantic model (dataset)
            query: DAX or MDX query to execute
            query_language: Query language ("DAX" or "MDX")
            dataset_name: Optional name for logging
        
        Returns:
            Query results with schema and data
        """
        if not FABRIC_DATA_AGENTS_ENABLED:
            raise ValueError("Fabric Data Agents are not enabled")
        
        logger.info(f"Querying semantic model {dataset_name or dataset_id} with {query_language}: {query[:100]}...")
        
        endpoint = f"workspaces/{self.workspace_id}/datasets/{dataset_id}/executeQueries"
        payload = {
            "queries": [
                {
                    "query": query
                }
            ],
            "serializerSettings": {
                "includeNulls": True
            },
            "impersonatedUserName": None
        }
        
        result = self._make_request("POST", endpoint, data=payload)
        
        logger.info(f"Semantic model query completed successfully")
        return result
    
    def list_lakehouses(self) -> List[Dict[str, Any]]:
        """List all lakehouses in the workspace"""
        endpoint = f"workspaces/{self.workspace_id}/lakehouses"
        result = self._make_request("GET", endpoint)
        return result.get("value", [])
    
    def list_warehouses(self) -> List[Dict[str, Any]]:
        """List all warehouses in the workspace"""
        endpoint = f"workspaces/{self.workspace_id}/warehouses"
        result = self._make_request("GET", endpoint)
        return result.get("value", [])
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all data pipelines in the workspace"""
        endpoint = f"workspaces/{self.workspace_id}/datapipelines"
        result = self._make_request("GET", endpoint)
        return result.get("value", [])
    
    def list_semantic_models(self) -> List[Dict[str, Any]]:
        """List all semantic models (datasets) in the workspace"""
        endpoint = f"workspaces/{self.workspace_id}/datasets"
        result = self._make_request("GET", endpoint)
        return result.get("value", [])


# =========================================
# Global Fabric API Client Instance
# =========================================

_fabric_client: Optional[FabricAPIClient] = None


def get_fabric_client() -> FabricAPIClient:
    """Get or create the global Fabric API client instance"""
    global _fabric_client
    
    if not FABRIC_ENABLED:
        raise ValueError("Fabric is not enabled. Set FABRIC_ENABLED=true to enable Fabric integration.")
    
    if not FABRIC_DATA_AGENTS_ENABLED:
        raise ValueError("Fabric Data Agents are not enabled. Set FABRIC_DATA_AGENTS_ENABLED=true to enable.")
    
    if not FABRIC_WORKSPACE_ID:
        raise ValueError("Fabric workspace ID is not configured. Set FABRIC_WORKSPACE_ID environment variable.")
    
    if _fabric_client is None:
        _fabric_client = FabricAPIClient()
        logger.info("Fabric API client initialized successfully")
    
    return _fabric_client


# =========================================
# MCP Tool Functions
# =========================================
# These functions will be decorated with @ai_function in next_best_action_agent.py

def fabric_query_lakehouse_tool(lakehouse_id: str, query: str, lakehouse_name: str = "") -> str:
    """
    Execute a Spark SQL query against a Fabric Lakehouse.
    
    This tool allows AI agents to query data in Fabric Lakehouses using Spark SQL.
    Use this for big data analytics, ETL operations, and data exploration.
    
    Args:
        lakehouse_id: The ID of the lakehouse to query
        query: The Spark SQL query to execute (e.g., "SELECT * FROM sales LIMIT 10")
        lakehouse_name: Optional friendly name of the lakehouse for logging
    
    Returns:
        JSON string containing query results with schema and data
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        result = client.query_lakehouse(lakehouse_id, query, lakehouse_name)
        
        return json.dumps({
            "success": True,
            "lakehouse_id": lakehouse_id,
            "lakehouse_name": lakehouse_name,
            "query": query,
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error querying lakehouse: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "lakehouse_id": lakehouse_id,
            "query": query
        })


def fabric_query_warehouse_tool(warehouse_id: str, query: str, warehouse_name: str = "") -> str:
    """
    Execute a T-SQL query against a Fabric Data Warehouse.
    
    This tool allows AI agents to query data in Fabric Data Warehouses using T-SQL.
    Use this for structured data analytics, reporting, and SQL-based operations.
    
    Args:
        warehouse_id: The ID of the warehouse to query
        query: The T-SQL query to execute (e.g., "SELECT TOP 10 * FROM customers")
        warehouse_name: Optional friendly name of the warehouse for logging
    
    Returns:
        JSON string containing query results with schema and data
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        result = client.query_warehouse(warehouse_id, query, warehouse_name)
        
        return json.dumps({
            "success": True,
            "warehouse_id": warehouse_id,
            "warehouse_name": warehouse_name,
            "query": query,
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error querying warehouse: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "warehouse_id": warehouse_id,
            "query": query
        })


def fabric_trigger_pipeline_tool(pipeline_id: str, pipeline_name: str = "", parameters: str = "{}") -> str:
    """
    Trigger execution of a Fabric Data Pipeline.
    
    This tool allows AI agents to start Fabric Data Pipelines for ETL, data movement,
    and orchestration operations.
    
    Args:
        pipeline_id: The ID of the pipeline to trigger
        pipeline_name: Optional friendly name of the pipeline for logging
        parameters: JSON string of parameters to pass to the pipeline (default: empty dict)
    
    Returns:
        JSON string containing pipeline run information including run ID
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        
        # Parse parameters JSON
        params_dict = json.loads(parameters) if parameters else {}
        
        result = client.trigger_pipeline(pipeline_id, pipeline_name, params_dict)
        
        return json.dumps({
            "success": True,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "run_id": result.get("runId"),
            "status": result.get("status"),
            "parameters": params_dict,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error triggering pipeline: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "pipeline_id": pipeline_id
        })


def fabric_get_pipeline_status_tool(pipeline_id: str, run_id: str, pipeline_name: str = "") -> str:
    """
    Get the status of a Fabric Data Pipeline run.
    
    This tool allows AI agents to monitor the execution status of Fabric Data Pipelines.
    Use this to check if a pipeline has completed, failed, or is still running.
    
    Args:
        pipeline_id: The ID of the pipeline
        run_id: The ID of the pipeline run to check
        pipeline_name: Optional friendly name of the pipeline for logging
    
    Returns:
        JSON string containing pipeline run status and details
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        result = client.get_pipeline_status(pipeline_id, run_id, pipeline_name)
        
        return json.dumps({
            "success": True,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "run_id": run_id,
            "status": result.get("status"),
            "start_time": result.get("startTime"),
            "end_time": result.get("endTime"),
            "duration": result.get("duration"),
            "details": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "pipeline_id": pipeline_id,
            "run_id": run_id
        })


def fabric_query_semantic_model_tool(
    dataset_id: str,
    query: str,
    dataset_name: str = "",
    query_language: str = "DAX"
) -> str:
    """
    Query a Power BI semantic model (dataset) using DAX or MDX.
    
    This tool allows AI agents to query Power BI semantic models for analytics
    and reporting. Supports both DAX (Data Analysis Expressions) and MDX queries.
    
    Args:
        dataset_id: The ID of the semantic model (dataset) to query
        query: The DAX or MDX query to execute
        dataset_name: Optional friendly name of the dataset for logging
        query_language: Query language to use ("DAX" or "MDX", default: "DAX")
    
    Returns:
        JSON string containing query results with schema and data
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        result = client.query_semantic_model(dataset_id, query, query_language, dataset_name)
        
        return json.dumps({
            "success": True,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "query": query,
            "query_language": query_language,
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error querying semantic model: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "dataset_id": dataset_id,
            "query": query
        })


def fabric_list_resources_tool(resource_type: str = "all") -> str:
    """
    List Fabric resources in the workspace.
    
    This tool allows AI agents to discover available Fabric resources
    (lakehouses, warehouses, pipelines, semantic models) in the workspace.
    
    Args:
        resource_type: Type of resources to list ("lakehouse", "warehouse", "pipeline",
                      "semantic_model", or "all" for all types)
    
    Returns:
        JSON string containing list of resources
    """
    if not FABRIC_DATA_AGENTS_ENABLED:
        return json.dumps({"error": "Fabric Data Agents are not enabled"})
    
    try:
        client = get_fabric_client()
        resources = {}
        
        if resource_type in ["all", "lakehouse"]:
            resources["lakehouses"] = client.list_lakehouses()
        
        if resource_type in ["all", "warehouse"]:
            resources["warehouses"] = client.list_warehouses()
        
        if resource_type in ["all", "pipeline"]:
            resources["pipelines"] = client.list_pipelines()
        
        if resource_type in ["all", "semantic_model"]:
            resources["semantic_models"] = client.list_semantic_models()
        
        return json.dumps({
            "success": True,
            "workspace_id": FABRIC_WORKSPACE_ID,
            "resource_type": resource_type,
            "resources": resources,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error listing Fabric resources: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "resource_type": resource_type
        })
