#!/usr/bin/env python3
"""
Ingest Prior Authorization policy documents into Azure AI Search.

Reads markdown files from task_instructions/, chunks them by heading sections,
generates embeddings with text-embedding-3-large, and uploads to the
'prior-authorization' search index.

Optionally ingests the CMS Interoperability & Prior Authorization Final Rule
(CMS-0057-F) web content as an additional knowledge source.

Usage:
    # Ingest local policies only:
    python scripts/ingest_prior_auth_policies.py

    # Include CMS website content:
    python scripts/ingest_prior_auth_policies.py --include-cms

    # Dry-run (print chunks, skip upload):
    python scripts/ingest_prior_auth_policies.py --dry-run

Environment variables:
    AZURE_SEARCH_ENDPOINT   - AI Search endpoint
    FOUNDRY_PROJECT_ENDPOINT - Azure AI Foundry endpoint (for embeddings)
    EMBEDDING_MODEL_DEPLOYMENT_NAME - Embedding model deployment (default: text-embedding-3-large)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_DEPLOYMENT_NAME", "text-embedding-3-large")
INDEX_NAME = os.getenv("AZURE_SEARCH_PA_INDEX_NAME", "prior-authorization")
EMBEDDING_DIMENSIONS = 3072
CHUNK_MAX_TOKENS = 512  # approximate max tokens per chunk

TASK_INSTRUCTIONS_DIR = Path(__file__).resolve().parent.parent / "task_instructions"

# Knowledge Base / Source names for agentic retrieval
PA_KNOWLEDGE_SOURCE_NAME = os.getenv("PA_KNOWLEDGE_SOURCE_NAME", "prior-authorization-source")
PA_KNOWLEDGE_BASE_NAME = os.getenv("PA_KNOWLEDGE_BASE_NAME", "prior-authorization-kb")

# CMS-0057-F content (structured for ingestion as knowledge source)
CMS_PRIOR_AUTH_CONTENT = {
    "source": "cms.gov",
    "source_type": "regulatory",
    "policy_id": "CMS-0057-F",
    "line_of_business": "Medicare Advantage",
    "effective_date": "2024-01-17",
    "version": "Final Rule",
    "sections": [
        {
            "title": "CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F) Overview",
            "content": (
                "The Centers for Medicare & Medicaid Services (CMS) released the CMS Interoperability "
                "and Prior Authorization Final Rule (CMS-0057-F) on January 17, 2024. This final rule "
                "emphasizes the need to improve health information exchange to achieve appropriate and "
                "necessary access to health records for patients, healthcare providers, and payers. "
                "This final rule also focuses on efforts to improve prior authorization processes through "
                "policies and technology, to help ensure that patients remain at the center of their own care. "
                "The rule enhances certain policies from the CMS Interoperability and Patient Access Final "
                "Rule (CMS-9115-F) and adds several new provisions to increase data sharing and reduce "
                "overall payer, healthcare provider, and patient burden through improvements to prior "
                "authorization practices and data exchange practices."
            ),
        },
        {
            "title": "CMS-0057-F Implementation Timelines",
            "content": (
                "Impacted payers are required to implement certain provisions by January 1, 2026. "
                "However, in response to stakeholder comments on the proposed rule, impacted payers "
                "have until primarily January 1, 2027, to meet the application programming interface "
                "(API) requirements in this final rule. The rule applies to Medicare Advantage "
                "organizations, state Medicaid and CHIP agencies, Medicaid managed care plans, CHIP "
                "managed care entities, and qualified health plan (QHP) issuers on Federally-Facilitated "
                "Exchanges (FFEs)."
            ),
        },
        {
            "title": "CMS-0057-F Prior Authorization API Requirements",
            "content": (
                "The Prior Authorization Requirements, Documentation, and Decision (PARDD) API requires "
                "impacted payers to build and maintain a FHIR-based Prior Authorization API that: "
                "1) Automates the determination of whether prior authorization is required for items and services. "
                "2) Identifies documentation requirements and any rules applied in making prior authorization decisions. "
                "3) Facilitates a prior authorization request and provides the response, including the specific "
                "reason for denial. Payers must respond to standard prior authorization requests within "
                "7 calendar days and to urgent (expedited) requests within 72 hours. "
                "Payers must also publicly report prior authorization metrics including approval rates, "
                "denial rates, average processing times, and overturn rates on appeal."
            ),
        },
        {
            "title": "CMS-0057-F Prior Authorization Decision Timelines",
            "content": (
                "Under the CMS-0057-F final rule, payers must meet specific turnaround times for prior "
                "authorization decisions: Standard non-urgent requests must receive a determination within "
                "7 calendar days of receipt. Urgent (expedited) requests where delay could seriously "
                "jeopardize the member's life, health, or ability to regain maximum function must receive "
                "a determination within 72 hours. These timelines apply to Medicare Advantage organizations "
                "and are intended to reduce administrative burden and ensure timely access to care."
            ),
        },
        {
            "title": "CMS-0057-F HIPAA Enforcement Discretion for FHIR APIs",
            "content": (
                "On February 28, 2024, the National Standards Group (NSG) announced an enforcement "
                "discretion for HIPAA covered entities that implement FHIR-based Prior Authorization "
                "APIs as described in the CMS Interoperability and Prior Authorization final rule. NSG "
                "will not take HIPAA Administrative Simplification enforcement action against HIPAA "
                "covered entities that choose not to use the X12 278 standard as part of an electronic "
                "FHIR prior authorization process."
            ),
        },
        {
            "title": "CMS-0057-F Continuity of Care and Interoperability Requirements",
            "content": (
                "The final rule includes provisions for payer-to-payer data exchange to support continuity "
                "of care when members transition between health plans. When a member transitions coverage, "
                "the prior payer must make available certain data including prior authorization decisions and "
                "supporting clinical documentation. This supports continuity-of-care considerations during "
                "coverage transitions regardless of provider network status. Ongoing treatment should not be "
                "disrupted solely due to plan transitions, and prior authorization approvals from a prior plan "
                "should be considered during transition periods."
            ),
        },
    ],
}


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def get_embedding(text: str, credential) -> List[float]:
    """Generate embeddings via Azure OpenAI text-embedding-3-large."""
    from openai import AzureOpenAI

    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    base_endpoint = (
        FOUNDRY_ENDPOINT.split("/api/projects")[0]
        if "/api/projects" in FOUNDRY_ENDPOINT
        else FOUNDRY_ENDPOINT
    )
    client = AzureOpenAI(
        azure_endpoint=base_endpoint,
        api_key=token.token,
        api_version="2024-02-15-preview",
    )
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Markdown parsing & chunking
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> Dict[str, str]:
    """Extract YAML-like frontmatter from markdown."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta


def chunk_markdown(filepath: Path, max_tokens: int = CHUNK_MAX_TOKENS) -> List[Dict[str, Any]]:
    """
    Chunk a markdown file by heading sections.
    Each chunk contains a section heading and its body text.
    Large sections are further split at paragraph boundaries.
    """
    raw = filepath.read_text(encoding="utf-8")
    meta = parse_frontmatter(raw)

    # Remove frontmatter
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, count=1, flags=re.DOTALL).strip()

    # Split by markdown headings
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: List[Dict[str, str]] = []
    last_end = 0
    last_title = meta.get("policy_id", filepath.stem)

    for m in heading_pattern.finditer(body):
        if last_end < m.start():
            section_text = body[last_end:m.start()].strip()
            if section_text:
                sections.append({"title": last_title, "content": section_text})
        last_title = m.group(2).strip()
        last_end = m.end()

    # Remaining text after last heading
    remaining = body[last_end:].strip()
    if remaining:
        sections.append({"title": last_title, "content": remaining})

    # Sub-chunk large sections
    chunks = []
    for sec in sections:
        content = sec["content"]
        approx_tokens = len(content.split())
        if approx_tokens <= max_tokens:
            chunks.append({
                "title": sec["title"],
                "content": content,
                **meta,
            })
        else:
            paragraphs = re.split(r"\n\n+", content)
            buf = ""
            for para in paragraphs:
                if buf and len((buf + "\n\n" + para).split()) > max_tokens:
                    chunks.append({"title": sec["title"], "content": buf.strip(), **meta})
                    buf = para
                else:
                    buf = (buf + "\n\n" + para).strip() if buf else para
            if buf.strip():
                chunks.append({"title": sec["title"], "content": buf.strip(), **meta})

    return chunks


def make_doc_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic document ID."""
    raw = f"{source}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def ensure_index(index_client: SearchIndexClient):
    """Create the search index if it doesn't exist."""
    try:
        index_client.get_index(INDEX_NAME)
        logger.info(f"Index '{INDEX_NAME}' already exists.")
        return
    except Exception:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="policy_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="effective_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="line_of_business", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="metadata", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-algorithm"),
        ],
        profiles=[
            VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-algorithm"),
        ],
    )

    semantic_config = SemanticConfiguration(
        name="semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            title_field=SemanticField(field_name="title"),
        ),
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    index_client.create_index(index)
    logger.info(f"Created index '{INDEX_NAME}'.")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_markdown_files(
    search_client: SearchClient,
    credential,
    dry_run: bool = False,
) -> int:
    """Chunk and ingest all MA-*.md files from task_instructions/."""
    total = 0
    md_files = sorted(TASK_INSTRUCTIONS_DIR.glob("MA-*.md"))
    if not md_files:
        logger.warning(f"No MA-*.md files found in {TASK_INSTRUCTIONS_DIR}")
        return 0

    for filepath in md_files:
        logger.info(f"Processing {filepath.name}")
        chunks = chunk_markdown(filepath)
        logger.info(f"  {len(chunks)} chunks")

        # Determine source_type from filename / content
        fname_lower = filepath.stem.lower()
        if "exception" in fname_lower:
            source_type = "exception_logic"
        elif "operational" in fname_lower or "guidance" in fname_lower:
            source_type = "operations"
        elif "coverage" in fname_lower or "policy" in fname_lower:
            source_type = "coverage_policy"
        else:
            source_type = "coverage_policy"

        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = make_doc_id(filepath.name, i)
            embed_text = f"{chunk['title']}\n{chunk['content']}"

            doc = {
                "id": doc_id,
                "content": chunk["content"],
                "title": chunk["title"],
                "source": filepath.name,
                "source_type": source_type,
                "policy_id": chunk.get("policy_id", ""),
                "effective_date": chunk.get("effective_date", ""),
                "version": chunk.get("version", ""),
                "line_of_business": chunk.get("line_of_business", ""),
                "chunk_index": i,
                "metadata": json.dumps({k: v for k, v in chunk.items() if k not in ("title", "content")}),
            }

            if dry_run:
                logger.info(f"  [DRY-RUN] chunk {i}: {chunk['title'][:60]}")
            else:
                embedding = get_embedding(embed_text, credential)
                doc["embedding"] = embedding
                # Rate-limit embedding calls
                time.sleep(0.1)

            documents.append(doc)

        if not dry_run and documents:
            result = search_client.upload_documents(documents)
            succeeded = sum(1 for r in result if r.succeeded)
            logger.info(f"  Uploaded {succeeded}/{len(documents)} documents from {filepath.name}")

        total += len(documents)

    return total


def ingest_cms_content(
    search_client: SearchClient,
    credential,
    dry_run: bool = False,
) -> int:
    """Ingest CMS-0057-F prior authorization rule content."""
    documents = []
    cms = CMS_PRIOR_AUTH_CONTENT

    for i, section in enumerate(cms["sections"]):
        doc_id = make_doc_id(f"cms-{cms['policy_id']}", i)
        embed_text = f"{section['title']}\n{section['content']}"

        doc = {
            "id": doc_id,
            "content": section["content"],
            "title": section["title"],
            "source": cms["source"],
            "source_type": cms["source_type"],
            "policy_id": cms["policy_id"],
            "effective_date": cms["effective_date"],
            "version": cms["version"],
            "line_of_business": cms["line_of_business"],
            "chunk_index": i,
            "metadata": json.dumps({
                "source_url": "https://www.cms.gov/priorities/burden-reduction/overview/interoperability/policies-regulations/cms-interoperability-prior-authorization-final-rule-cms-0057-f",
                "rule_id": cms["policy_id"],
            }),
        }

        if dry_run:
            logger.info(f"  [DRY-RUN] CMS chunk {i}: {section['title'][:60]}")
        else:
            embedding = get_embedding(embed_text, credential)
            doc["embedding"] = embedding
            time.sleep(0.1)

        documents.append(doc)

    if not dry_run and documents:
        result = search_client.upload_documents(documents)
        succeeded = sum(1 for r in result if r.succeeded)
        logger.info(f"  Uploaded {succeeded}/{len(documents)} CMS documents")

    return len(documents)


# ---------------------------------------------------------------------------
# Knowledge Source / Knowledge Base provisioning (agentic retrieval)
# ---------------------------------------------------------------------------

def _get_search_token(credential) -> str:
    """Acquire bearer token for Azure AI Search scope."""
    token = credential.get_token("https://search.azure.com/.default")
    return token.token


def ensure_knowledge_source(credential) -> None:
    """
    Create Knowledge Source and Knowledge Base for the prior-authorization index.
    Uses the 2025-11-01-preview agentic retrieval REST API.
    """
    import requests as _requests

    try:
        if not SEARCH_ENDPOINT:
            logger.warning("Knowledge Source provisioning skipped: AZURE_SEARCH_ENDPOINT not set")
            return

        search_base = SEARCH_ENDPOINT.rstrip("/")
        api_version = "2025-11-01-preview"
        headers = {
            "Authorization": f"Bearer {_get_search_token(credential)}",
            "Content-Type": "application/json",
        }

        # ── 1. Create Knowledge Source ──────────────────────────
        ks_url = f"{search_base}/knowledgesources/{PA_KNOWLEDGE_SOURCE_NAME}?api-version={api_version}"
        resp = _requests.get(ks_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            logger.info(f"Knowledge Source already exists: {PA_KNOWLEDGE_SOURCE_NAME}")
        else:
            ks_payload = {
                "name": PA_KNOWLEDGE_SOURCE_NAME,
                "kind": "searchIndex",
                "description": "Prior authorization policies knowledge source for agentic retrieval",
                "searchIndexParameters": {
                    "searchIndexName": INDEX_NAME,
                    "semanticConfigurationName": "semantic-config",
                    "sourceDataFields": [
                        {"name": "id"},
                        {"name": "title"},
                        {"name": "content"},
                        {"name": "source_type"},
                        {"name": "policy_id"},
                    ],
                },
            }
            create_resp = _requests.put(ks_url, headers=headers, json=ks_payload, timeout=30)
            if create_resp.status_code in (200, 201):
                logger.info(f"Knowledge Source created: {PA_KNOWLEDGE_SOURCE_NAME}")
            else:
                logger.warning(
                    f"Knowledge Source create returned {create_resp.status_code}: {create_resp.text}"
                )
                return

        # ── 2. Create Knowledge Base ────────────────────────────
        kb_url = f"{search_base}/knowledgebases/{PA_KNOWLEDGE_BASE_NAME}?api-version={api_version}"
        resp = _requests.get(kb_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            logger.info(f"Knowledge Base already exists: {PA_KNOWLEDGE_BASE_NAME}")
            return

        kb_payload = {
            "name": PA_KNOWLEDGE_BASE_NAME,
            "description": "Prior authorization knowledge base for agentic retrieval",
            "knowledgeSources": [
                {"name": PA_KNOWLEDGE_SOURCE_NAME},
            ],
        }
        # Add model config for agentic retrieval reasoning (effort > minimal)
        if FOUNDRY_ENDPOINT:
            base_endpoint = (
                FOUNDRY_ENDPOINT.split("/api/projects")[0]
                if "/api/projects" in FOUNDRY_ENDPOINT
                else FOUNDRY_ENDPOINT
            )
            kb_payload["models"] = [{
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                    "resourceUri": base_endpoint,
                    "deploymentId": EMBEDDING_MODEL.replace("text-embedding-3-large", "gpt-4o-mini"),
                    "modelName": "gpt-4o-mini",
                },
            }]
        create_resp = _requests.put(kb_url, headers=headers, json=kb_payload, timeout=30)
        if create_resp.status_code in (200, 201):
            logger.info(f"Knowledge Base created: {PA_KNOWLEDGE_BASE_NAME}")
        else:
            logger.warning(
                f"Knowledge Base create returned {create_resp.status_code}: {create_resp.text}"
            )
    except Exception as ex:
        logger.warning(f"Knowledge Source / Base provisioning skipped: {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest prior-auth policies into AI Search")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk without uploading")
    parser.add_argument("--include-cms", action="store_true", help="Include CMS-0057-F web content")
    args = parser.parse_args()

    if not args.dry_run:
        if not SEARCH_ENDPOINT:
            logger.error("AZURE_SEARCH_ENDPOINT not set")
            sys.exit(1)
        if not FOUNDRY_ENDPOINT:
            logger.error("FOUNDRY_PROJECT_ENDPOINT not set")
            sys.exit(1)

    credential = DefaultAzureCredential() if not args.dry_run else None

    if not args.dry_run:
        index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)
        ensure_index(index_client)
        search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)
    else:
        search_client = None

    total = 0

    # 1. Ingest local markdown policies
    logger.info("=== Ingesting local policy documents ===")
    total += ingest_markdown_files(search_client, credential, dry_run=args.dry_run)

    # 2. Optionally ingest CMS web content
    if args.include_cms:
        logger.info("=== Ingesting CMS-0057-F content ===")
        total += ingest_cms_content(search_client, credential, dry_run=args.dry_run)

    # 3. Provision Knowledge Source + Knowledge Base for agentic retrieval
    if not args.dry_run:
        logger.info("=== Provisioning Knowledge Source & Knowledge Base ===")
        ensure_knowledge_source(credential)

    logger.info(f"=== Done. Total chunks: {total} ===")


if __name__ == "__main__":
    main()
