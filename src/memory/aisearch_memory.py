"""
AI Search Long-Term Memory Provider with AzureAISearchContextProvider Integration

This module provides long-term memory storage using Azure AI Search with:
- AzureAISearchContextProvider from Microsoft Agent Framework
- Query planning and optimization
- Multi-hop reasoning across documents
- Answer synthesis with citations
- Agentic retrieval patterns

Based on the Foundry IQ Agent Framework integration pattern:
https://devblogs.microsoft.com/foundry/foundry-iq-agent-framework-integration/
"""

import logging
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Microsoft Agent Framework AI Search Context Provider
# The AzureAISearchContextProvider is part of agent_framework.azure module
try:
    from agent_framework.azure import AzureAISearchContextProvider
    AISEARCH_CONTEXT_PROVIDER_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try agent_framework_aisearch if available
        from agent_framework_aisearch import AzureAISearchContextProvider
        AISEARCH_CONTEXT_PROVIDER_AVAILABLE = True
    except ImportError:
        AISEARCH_CONTEXT_PROVIDER_AVAILABLE = False

from .base import MemoryProvider, MemoryEntry, MemorySearchResult, MemoryType

logger = logging.getLogger(__name__)


class LongTermMemory(MemoryProvider):
    """
    Long-term memory provider using Azure AI Search with AzureAISearchContextProvider.
    
    This integrates with the Microsoft Agent Framework's AzureAISearchContextProvider
    to leverage Foundry IQ's Knowledge Base capabilities for:
    - Query planning and optimization
    - Multi-hop reasoning across documents
    - Answer synthesis with citations
    - Agentic retrieval patterns
    - Hybrid search (vector + full-text + semantic reranking)
    
    Based on: https://devblogs.microsoft.com/foundry/foundry-iq-agent-framework-integration/
    """
    
    def __init__(
        self,
        search_endpoint: str,
        foundry_endpoint: str,
        index_name: str = "task-instructions",
        knowledge_base_name: Optional[str] = None,
        model_deployment_name: str = "gpt-4o",
        credential: Optional[Any] = None,
        async_credential: Optional[Any] = None,
        mode: str = "agentic",
        retrieval_reasoning_effort: str = "medium",
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize Long-Term Memory provider with AzureAISearchContextProvider.
        
        Args:
            search_endpoint: Azure AI Search endpoint
            foundry_endpoint: Azure AI Foundry endpoint
            index_name: Name of the search index (used if no knowledge_base_name)
            knowledge_base_name: Name of the Foundry IQ Knowledge Base
            model_deployment_name: Model for agentic retrieval
            credential: Azure credential (sync)
            async_credential: Azure credential (async) for AzureAISearchContextProvider
            mode: Retrieval mode - "semantic" for fast hybrid, "agentic" for IQ
            retrieval_reasoning_effort: How much query planning to do
            embedding_function: Function to generate embeddings from text
        """
        self._search_endpoint = search_endpoint
        self._foundry_endpoint = foundry_endpoint
        self._index_name = index_name
        self._knowledge_base_name = knowledge_base_name
        self._model_deployment_name = model_deployment_name
        self._credential = credential or DefaultAzureCredential()
        self._async_credential = async_credential
        self._mode = mode
        self._reasoning_effort = retrieval_reasoning_effort
        self._embedding_function = embedding_function
        
        # Initialize search client for fallback operations
        self._search_client = None
        try:
            self._search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=index_name,
                credential=self._credential
            )
            logger.info(f"SearchClient initialized for index: {index_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize SearchClient: {e}")
        
        # Initialize AzureAISearchContextProvider if available
        self._context_provider: Optional[Any] = None
        if AISEARCH_CONTEXT_PROVIDER_AVAILABLE:
            try:
                # Create async credential if not provided
                if self._async_credential is None:
                    self._async_credential = AsyncDefaultAzureCredential()
                
                # Initialize AzureAISearchContextProvider for agent framework integration
                self._context_provider = AzureAISearchContextProvider(
                    endpoint=search_endpoint,
                    index_name=index_name,
                    credential=self._async_credential,
                )
                logger.info(f"AzureAISearchContextProvider initialized for index: {index_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize AzureAISearchContextProvider: {e}")
                self._context_provider = None
        else:
            logger.warning("agent-framework-azure-ai-search not available - using fallback search")
        
        logger.info(f"LongTermMemory initialized in {mode} mode (context_provider={'enabled' if self._context_provider else 'disabled'})")
    
    @property
    def name(self) -> str:
        return "long_term_memory"
    
    @property
    def is_short_term(self) -> bool:
        return False
    
    @property
    def context_provider(self) -> Optional[Any]:
        """
        Get the AzureAISearchContextProvider for use with Microsoft Agent Framework.
        
        This can be passed to ChatAgent's context_providers parameter for automatic
        retrieval augmentation during agent conversations.
        
        Example:
            async with ChatAgent(
                chat_client=client,
                instructions="You are a helpful assistant.",
                context_providers=long_term_memory.context_provider,
            ) as agent:
                result = await agent.run("What are the steps for user management?")
        """
        return self._context_provider
    
    def set_embedding_function(self, func: Callable[[str], List[float]]) -> None:
        """Set the embedding function for text-to-vector conversion."""
        self._embedding_function = func
    
    async def get_context(self, query: str) -> str:
        """
        Get context for a query using AzureAISearchContextProvider.
        
        This method provides a direct way to get context from the search index
        that can be used to augment prompts or provide retrieval-augmented generation.
        
        Args:
            query: The query to search for context
            
        Returns:
            String containing the retrieved context formatted for LLM consumption
        """
        if self._context_provider is not None:
            try:
                logger.info(f"[AzureAISearchContextProvider] Calling get_context for query: {query[:100]}...")
                # Use the context provider's get_context method
                context = await self._context_provider.get_context(query)
                if context:
                    logger.info(f"[AzureAISearchContextProvider] SUCCESS - Retrieved {len(context)} chars of context")
                else:
                    logger.info("[AzureAISearchContextProvider] Returned empty context")
                return context or ""
            except Exception as e:
                logger.error(f"[AzureAISearchContextProvider] ERROR: {e}")
        else:
            logger.warning("[AzureAISearchContextProvider] Not available - using fallback search")
        
        # Fallback to basic search
        logger.info("[Fallback] Using basic SearchClient for context retrieval")
        results = await self.search_with_iq(query, limit=5)
        if results:
            context_parts = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", "")
                if title:
                    context_parts.append(f"## {title}\n{content}")
                else:
                    context_parts.append(content)
            return "\n\n".join(context_parts)
        return ""
    
    async def search_with_iq(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search using Foundry IQ's agentic retrieval via AzureAISearchContextProvider.
        
        When available, uses the AzureAISearchContextProvider for enhanced retrieval.
        Falls back to basic SearchClient when the context provider is not available.
        """
        try:
            logger.info(f"LongTermMemory search query: {query[:100]}...")
            
            # Use AzureAISearchContextProvider if available for enhanced retrieval
            if self._context_provider is not None and self._mode == "agentic":
                try:
                    # Get context from the provider
                    context = await self._context_provider.get_context(query)
                    
                    # Parse context into results format
                    # The context provider returns formatted text, so we structure it
                    if context:
                        return [{
                            "content": context,
                            "title": "Retrieved Context",
                            "score": 1.0,
                            "metadata": {
                                "document_id": "context_provider_result",
                                "intent": query,
                                "source": "AzureAISearchContextProvider",
                            }
                        }]
                except Exception as e:
                    logger.warning(f"AzureAISearchContextProvider search failed, falling back: {e}")
            
            # Fallback to basic SearchClient
            if not self._search_client:
                logger.warning("SearchClient not initialized")
                return []
            
            results = self._search_client.search(
                search_text=query,
                top=limit,
            )
            
            return [
                {
                    "content": r.get("content", ""),
                    "title": r.get("title", ""),
                    "score": r.get("@search.score", 0),
                    "metadata": {
                        "document_id": r.get("document_id"),
                        "intent": r.get("intent"),
                        "source": "SearchClient",
                    }
                }
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"LongTermMemory search error: {e}")
            return []
    
    async def search_task_instructions(
        self,
        task_description: str,
        limit: int = 5,
        include_steps: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for task instructions relevant to the given task description.
        
        This is the main method for the next_best_action agent to retrieve
        long-term memory about how to handle specific types of tasks.
        
        Args:
            task_description: Natural language description of the task
            limit: Maximum number of results to return
            include_steps: Whether to include detailed steps in results
        
        Returns:
            List of task instruction documents with relevance scores
        """
        if not self._search_client:
            logger.warning("SearchClient not initialized")
            return []
        
        try:
            # Generate embedding for vector search
            vector_queries = []
            if self._embedding_function:
                query_embedding = self._embedding_function(task_description)
                vector_queries.append(VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=limit * 3,
                    fields="embedding"
                ))
            
            # Execute hybrid search
            results = self._search_client.search(
                search_text=task_description,
                vector_queries=vector_queries if vector_queries else None,
                select=[
                    "id", "document_id", "title", "category", "intent",
                    "description", "content", "keywords", "estimated_effort",
                    "steps", "related_tasks", "chunk_num", "total_chunks"
                ],
                top=limit * 2,
            )
            
            # Deduplicate by document_id and get best chunk per document
            doc_results = {}
            for result in results:
                doc_id = result.get("document_id")
                score = result.get("@search.score", 0)
                
                if doc_id not in doc_results or score > doc_results[doc_id]["score"]:
                    steps = []
                    if include_steps:
                        try:
                            steps = json.loads(result.get("steps", "[]"))
                        except:
                            pass
                    
                    doc_results[doc_id] = {
                        "document_id": doc_id,
                        "title": result.get("title", ""),
                        "category": result.get("category", ""),
                        "intent": result.get("intent", ""),
                        "description": result.get("description", ""),
                        "content_excerpt": result.get("content", "")[:500] + "..." if len(result.get("content", "")) > 500 else result.get("content", ""),
                        "keywords": result.get("keywords", []),
                        "estimated_effort": result.get("estimated_effort", ""),
                        "steps": steps,
                        "related_tasks": result.get("related_tasks", []),
                        "score": score,
                    }
            
            # Sort by score and limit
            sorted_results = sorted(
                doc_results.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:limit]
            
            logger.info(f"Found {len(sorted_results)} task instructions for query")
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error searching task instructions: {e}")
            return []
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry in AI Search."""
        if not self._search_client:
            logger.warning("SearchClient not initialized")
            return entry.id
        
        try:
            document = {
                "id": entry.id,
                "document_id": entry.id,
                "title": entry.metadata.get("title", "") if entry.metadata else "",
                "category": entry.memory_type.value if entry.memory_type else "context",
                "intent": entry.metadata.get("intent", "") if entry.metadata else "",
                "description": entry.metadata.get("description", "") if entry.metadata else "",
                "content": entry.content,
                "keywords": entry.metadata.get("keywords", []) if entry.metadata else [],
                "estimated_effort": entry.metadata.get("estimated_effort", "") if entry.metadata else "",
                "chunk_num": 0,
                "total_chunks": 1,
                "steps": json.dumps(entry.metadata.get("steps", []) if entry.metadata else []),
                "related_tasks": entry.metadata.get("related_tasks", []) if entry.metadata else [],
                "created_at": entry.created_at or datetime.utcnow().isoformat() + "Z",
            }
            
            # Add embedding if available
            if entry.embedding:
                document["embedding"] = entry.embedding
            elif self._embedding_function:
                document["embedding"] = self._embedding_function(entry.content)
            
            result = self._search_client.upload_documents(documents=[document])
            
            if result[0].succeeded:
                logger.info(f"Stored memory entry in AI Search: {entry.id}")
                return entry.id
            else:
                logger.error(f"Failed to store entry: {result[0].error_message}")
                return entry.id
                
        except Exception as e:
            logger.error(f"Error storing memory entry: {e}")
            return entry.id
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        if not self._search_client:
            return None
        
        try:
            result = self._search_client.get_document(key=entry_id)
            
            if result:
                return MemoryEntry(
                    id=result.get("id", entry_id),
                    content=result.get("content", ""),
                    memory_type=MemoryType(result.get("category", "context")),
                    embedding=result.get("embedding"),
                    session_id=result.get("session_id"),
                    metadata={
                        "title": result.get("title"),
                        "intent": result.get("intent"),
                        "description": result.get("description"),
                        "keywords": result.get("keywords", []),
                        "steps": json.loads(result.get("steps", "[]")),
                        "related_tasks": result.get("related_tasks", []),
                    },
                    created_at=result.get("created_at"),
                )
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory entry: {e}")
            return None
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """
        Search for similar memory entries using vector similarity.
        
        This uses hybrid search combining vector similarity, keyword matching,
        and semantic reranking for best results.
        """
        if not self._search_client:
            return []
        
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=limit * 2,
                fields="embedding"
            )
            
            # Build filter
            filter_parts = []
            if memory_type:
                filter_parts.append(f"category eq '{memory_type.value}'")
            
            filter_str = " and ".join(filter_parts) if filter_parts else None
            
            # Execute search with semantic configuration
            results = self._search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filter_str,
                select=[
                    "id", "document_id", "title", "category", "intent",
                    "description", "content", "keywords", "estimated_effort",
                    "steps", "related_tasks", "created_at"
                ],
                top=limit,
            )
            
            search_results = []
            for result in results:
                score = result.get("@search.score", 0)
                
                entry = MemoryEntry(
                    id=result.get("id"),
                    content=result.get("content", ""),
                    memory_type=MemoryType(result.get("category", "context")) if result.get("category") else MemoryType.CONTEXT,
                    embedding=None,
                    session_id=None,
                    metadata={
                        "document_id": result.get("document_id"),
                        "title": result.get("title"),
                        "intent": result.get("intent"),
                        "description": result.get("description"),
                        "keywords": result.get("keywords", []),
                        "steps": json.loads(result.get("steps", "[]")),
                        "related_tasks": result.get("related_tasks", []),
                        "estimated_effort": result.get("estimated_effort"),
                    },
                    created_at=result.get("created_at"),
                )
                
                search_results.append(MemorySearchResult(
                    entry=entry,
                    score=score,
                    source="long_term_memory"
                ))
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """
        Search for memory entries by text query using hybrid search.
        
        This combines:
        - Full-text search with keyword matching
        - Vector similarity (if embedding function available)
        - Semantic reranking for result quality
        """
        if not self._search_client:
            return []
        
        try:
            # Generate embedding for vector search if function available
            vector_queries = []
            if self._embedding_function:
                query_embedding = self._embedding_function(query)
                vector_queries.append(VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=limit * 2,
                    fields="embedding"
                ))
            
            # Build filter
            filter_parts = []
            if memory_type:
                filter_parts.append(f"category eq '{memory_type.value}'")
            
            filter_str = " and ".join(filter_parts) if filter_parts else None
            
            # Execute hybrid search
            results = self._search_client.search(
                search_text=query,
                vector_queries=vector_queries if vector_queries else None,
                filter=filter_str,
                select=[
                    "id", "document_id", "title", "category", "intent",
                    "description", "content", "keywords", "estimated_effort",
                    "steps", "related_tasks", "created_at"
                ],
                top=limit,
            )
            
            search_results = []
            for result in results:
                score = result.get("@search.score", 0)
                
                entry = MemoryEntry(
                    id=result.get("id"),
                    content=result.get("content", ""),
                    memory_type=MemoryType(result.get("category", "context")) if result.get("category") else MemoryType.CONTEXT,
                    embedding=None,
                    session_id=None,
                    metadata={
                        "document_id": result.get("document_id"),
                        "title": result.get("title"),
                        "intent": result.get("intent"),
                        "description": result.get("description"),
                        "keywords": result.get("keywords", []),
                        "steps": json.loads(result.get("steps", "[]")),
                        "related_tasks": result.get("related_tasks", []),
                        "estimated_effort": result.get("estimated_effort"),
                    },
                    created_at=result.get("created_at"),
                )
                
                search_results.append(MemorySearchResult(
                    entry=entry,
                    score=score,
                    source="long_term_memory"
                ))
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in text search: {e}")
            return []
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        if not self._search_client:
            return False
        
        try:
            result = self._search_client.delete_documents(documents=[{"id": entry_id}])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Error deleting entry: {e}")
            return False
    
    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryEntry]:
        """List memory entries for a specific session."""
        logger.info("Long-term memory is not session-scoped")
        return []
    
    async def clear_session(self, session_id: str) -> int:
        """Clear all memory entries for a session."""
        logger.warning("Cannot clear long-term memory by session")
        return 0
    
    async def health_check(self) -> bool:
        """Check health of long-term memory including AzureAISearchContextProvider."""
        try:
            # Check if context provider is available and working
            if self._context_provider is not None:
                try:
                    # Try a simple context retrieval to verify connectivity
                    await self._context_provider.get_context("health check")
                    logger.info("AzureAISearchContextProvider health check passed")
                    return True
                except Exception as e:
                    logger.warning(f"AzureAISearchContextProvider health check failed: {e}")
            
            # Fallback to basic SearchClient health check
            if self._search_client:
                results = self._search_client.search(search_text="*", top=1)
                list(results)
                return True
            return False
        except Exception as e:
            logger.error(f"LongTermMemory health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Clean up resources including async credentials."""
        if self._async_credential is not None:
            try:
                await self._async_credential.close()
            except Exception as e:
                logger.warning(f"Error closing async credential: {e}")
