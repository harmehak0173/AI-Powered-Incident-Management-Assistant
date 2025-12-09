"""
Vector Store Implementation for RAG

Supports both simple numpy-based similarity search and ChromaDB for production.
Uses sentence-transformers for embeddings.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Try to import sentence_transformers, fall back to simple hashing if not available
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class Document:
    """A document in the vector store"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """A search result from the vector store"""
    document: Document
    score: float
    highlights: List[str] = None
    
    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


class VectorStore:
    """
    In-memory vector store with cosine similarity search.
    
    In production, this would use Pinecone, Weaviate, ChromaDB, or similar.
    Supports sentence-transformers for real embeddings or simple hash-based 
    embeddings for demo purposes.
    """
    
    def __init__(self, use_real_embeddings: bool = False, embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents: List[Document] = []
        self.initialized = False
        self.use_real_embeddings = use_real_embeddings and EMBEDDINGS_AVAILABLE
        self.embedding_dim = 384 if self.use_real_embeddings else 128
        
        if self.use_real_embeddings:
            print(f"Loading embedding model: {embedding_model}")
            self.model = SentenceTransformer(embedding_model)
        else:
            self.model = None
            print("Using simple hash-based embeddings (install sentence-transformers for better results)")
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text"""
        if self.use_real_embeddings and self.model:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            return self._simple_hash_embedding(text)
    
    def _simple_hash_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based embedding for demo purposes"""
        embedding = np.zeros(128)
        normalized = text.lower()
        words = normalized.split()
        
        for idx, word in enumerate(words):
            for i, char in enumerate(word):
                char_code = ord(char)
                embedding[(char_code + idx) % 128] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> Document:
        """Add a document to the vector store"""
        embedding = self._compute_embedding(content)
        doc = Document(
            id=doc_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.documents.append(doc)
        return doc
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Add multiple documents"""
        added = []
        for doc in documents:
            added.append(self.add_document(
                doc_id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {})
            ))
        return added
    
    def search(
        self,
        query: str,
        limit: int = 10,
        doc_types: Optional[List[str]] = None,
        min_score: float = 0.1
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            limit: Maximum number of results
            doc_types: Filter by document type (from metadata)
            min_score: Minimum similarity score
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        query_embedding = self._compute_embedding(query)
        
        results = []
        for doc in self.documents:
            # Filter by type if specified
            if doc_types and doc.metadata.get("type") not in doc_types:
                continue
            
            score = self._cosine_similarity(query_embedding, doc.embedding)
            
            if score >= min_score:
                highlights = self._extract_highlights(doc.content, query)
                results.append(SearchResult(
                    document=doc,
                    score=float(score),
                    highlights=highlights
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]
    
    def search_similar(
        self,
        doc_id: str,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """Find documents similar to an existing document"""
        # Find the source document
        source_doc = None
        for doc in self.documents:
            if doc.id == doc_id:
                source_doc = doc
                break
        
        if not source_doc:
            return []
        
        results = []
        for doc in self.documents:
            if exclude_self and doc.id == doc_id:
                continue
            
            score = self._cosine_similarity(source_doc.embedding, doc.embedding)
            results.append(SearchResult(
                document=doc,
                score=float(score),
                highlights=[]
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def _extract_highlights(self, content: str, query: str, max_highlights: int = 3) -> List[str]:
        """Extract relevant snippets from content that match query terms"""
        highlights = []
        query_words = set(query.lower().split())
        sentences = content.replace('\n', ' ').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words if len(word) > 3):
                highlights.append(sentence)
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        type_counts = {}
        for doc in self.documents:
            doc_type = doc.metadata.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "by_type": type_counts,
            "embedding_dim": self.embedding_dim,
            "using_real_embeddings": self.use_real_embeddings
        }
    
    def clear(self):
        """Clear all documents from the store"""
        self.documents = []
        self.initialized = False


class IncidentVectorStore(VectorStore):
    """
    Specialized vector store for incident management data.
    Pre-configured to index incidents, playbooks, deployments, and alerts.
    """
    
    def initialize_from_data(
        self,
        incidents: List[Any],
        playbooks: List[Any],
        deployments: List[Any],
        alerts: List[Any]
    ):
        """Initialize the store with incident management data"""
        if self.initialized:
            return
        
        # Index incidents
        for incident in incidents:
            content = f"""
            Incident: {incident.title}
            Description: {incident.description}
            Severity: {incident.severity.value}
            Status: {incident.status.value}
            Services: {', '.join(incident.services)}
            Tags: {', '.join(incident.tags)}
            {f'Root Cause: {incident.root_cause}' if incident.root_cause else ''}
            {f'Resolution: {incident.resolution}' if incident.resolution else ''}
            Timeline: {' '.join(e.content for e in incident.timeline)}
            """.strip()
            
            self.add_document(
                doc_id=f"incident-{incident.id}",
                content=content,
                metadata={
                    "type": "incident",
                    "source_id": incident.id,
                    "title": incident.title,
                    "severity": incident.severity.value,
                    "status": incident.status.value,
                    "services": incident.services,
                    "created_at": incident.created_at.isoformat()
                }
            )
        
        # Index playbooks
        for playbook in playbooks:
            steps_text = ' '.join(f"{s.title}: {s.description}" for s in playbook.steps)
            content = f"""
            Playbook: {playbook.title}
            Description: {playbook.description}
            Services: {', '.join(playbook.services)}
            Tags: {', '.join(playbook.tags)}
            Steps: {steps_text}
            """.strip()
            
            self.add_document(
                doc_id=f"playbook-{playbook.id}",
                content=content,
                metadata={
                    "type": "playbook",
                    "source_id": playbook.id,
                    "title": playbook.title,
                    "services": playbook.services,
                    "usage_count": playbook.usage_count
                }
            )
        
        # Index deployments
        for deployment in deployments:
            content = f"""
            Deployment: {deployment.service} v{deployment.version}
            Environment: {deployment.environment.value}
            Status: {deployment.status.value}
            Commit: {deployment.commit_message}
            Changed Files: {', '.join(deployment.changed_files)}
            Deployed by: {deployment.deployed_by}
            """.strip()
            
            self.add_document(
                doc_id=f"deployment-{deployment.id}",
                content=content,
                metadata={
                    "type": "deployment",
                    "source_id": deployment.id,
                    "title": f"{deployment.service} v{deployment.version}",
                    "service": deployment.service,
                    "deployed_at": deployment.deployed_at.isoformat()
                }
            )
        
        # Index alerts
        for alert in alerts:
            content = f"""
            Alert: {alert.title}
            Description: {alert.description}
            Severity: {alert.severity.value}
            Service: {alert.service}
            Source: {alert.source}
            Status: {alert.status.value}
            """.strip()
            
            self.add_document(
                doc_id=f"alert-{alert.id}",
                content=content,
                metadata={
                    "type": "alert",
                    "source_id": alert.id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "service": alert.service
                }
            )
        
        self.initialized = True
        print(f"Vector store initialized with {len(self.documents)} documents")
    
    def find_similar_incidents(self, incident_id: str, limit: int = 5) -> List[SearchResult]:
        """Find incidents similar to a given incident"""
        doc_id = f"incident-{incident_id}"
        results = self.search_similar(doc_id, limit=limit + 1, exclude_self=True)
        # Filter to only incidents
        return [r for r in results if r.document.metadata.get("type") == "incident"][:limit]
    
    def find_relevant_playbooks(self, incident_title: str, services: List[str]) -> List[SearchResult]:
        """Find playbooks relevant to an incident"""
        query = f"{incident_title} {' '.join(services)}"
        return self.search(query, limit=3, doc_types=["playbook"])
    
    def find_recent_deployments(self, services: List[str], limit: int = 5) -> List[SearchResult]:
        """Find recent deployments for given services"""
        query = ' '.join(services)
        return self.search(query, limit=limit, doc_types=["deployment"])


# Global instance
_vector_store: Optional[IncidentVectorStore] = None


def get_vector_store() -> IncidentVectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = IncidentVectorStore(use_real_embeddings=False)
    return _vector_store


def initialize_vector_store(incidents, playbooks, deployments, alerts):
    """Initialize the global vector store with data"""
    store = get_vector_store()
    store.initialize_from_data(incidents, playbooks, deployments, alerts)
    return store
