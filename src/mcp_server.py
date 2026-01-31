"""
FastAPI MCP Server - Replacement for Azure Functions
Implements Model Context Protocol (MCP) with SSE support
"""

import json
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server for AI Agents",
    version="1.0.0"
)

# Azure Storage configuration
STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

# Initialize storage client
if STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
elif STORAGE_ACCOUNT_URL:
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=credential)
else:
    logger.warning("No storage configuration found - snippet storage will not work")
    blob_service_client = None

SNIPPETS_CONTAINER = "snippets"

# In-memory session storage (replace with Redis for production)
sessions: Dict[str, Dict[str, Any]] = {}


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
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
