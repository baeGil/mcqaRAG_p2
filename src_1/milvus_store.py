"""
Simplified Milvus Vector Store with HNSW indexing and hybrid search.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

from src.config import config
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Collection, MilvusException, connections, db, utility
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

class MilvusStore:
    """Simplified Milvus store with HNSW indexing and hybrid search."""
    
    def __init__(
        self, 
        uri: str = None,
        db_name: str = None,
        collection_name: str = None,
        embed_model: str = None,
        drop_old: bool = False
    ):
        """Initialize simplified MilvusStore."""
        self.uri = uri or config.get("database", "uri", default="http://localhost:19530")
        self.db_name = db_name or config.get("database", "name", default="gil")
        self.collection_name = collection_name or config.get("database", "collection_name", default="multimodal_rag")
        self.embed_model = embed_model or config.get("model", "embeddings", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        
        self._connect_to_milvus()
        self._initialize_database(drop_old=drop_old)
        
        # Create embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(model_name=self.embed_model)
        
        # Create vector store with HNSW index
        self.vector_store = self._create_vector_store_with_hnsw(drop_old=drop_old)
    
    def _connect_to_milvus(self) -> None:
        """Connect to local Milvus server."""
        host = self.uri.split("://")[1].split(":")[0]
        port = int(self.uri.split(":")[-1])
        connections.connect(host=host, port=port)
    
    def _initialize_database(self, drop_old: bool = False) -> None:
        """Initialize database."""
        try:
            existing_databases = db.list_database()
            if self.db_name not in existing_databases:
                db.create_database(self.db_name)
                logger.info(f"Database '{self.db_name}' created")
            
            db.using_database(self.db_name)
            
            if drop_old:
                collections = utility.list_collections()
                if self.collection_name in collections:
                    Collection(self.collection_name).drop()
                    logger.info(f"Collection '{self.collection_name}' dropped")
        except MilvusException as e:
            logger.error(f"Database initialization error: {e}")
    
    def _create_vector_store_with_hnsw(self, drop_old: bool = False) -> Milvus:
        """Create vector store with HNSW index configuration."""
        connection_args = {
            "uri": self.uri,
            "db_name": self.db_name
        }
        
        # HNSW index parameters based on documentation [(1)](https://milvus.io/docs/index.md#Indexes-supported-in-Milvus)
        index_params = [
            {
                "field_name": "dense",
                "index_type": "HNSW", 
                "metric_type": "COSINE",
                "params": {
                    "M": 48,
                    "efConstruction": 512
                }
            },
            {
                "field_name": "sparse",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "params": {
                    "term_frequency_enabled": True
                }
            }
        ]

        return Milvus(
            embedding_function=self.embeddings_model,
            connection_args=connection_args,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            drop_old=drop_old,
            collection_name=self.collection_name,
            auto_id=True,
            index_params=index_params  # Apply HNSW index
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        if not documents:
            logger.warning("No documents to add")
            return []
        
        try:
            ids = self.vector_store.add_documents(documents=documents)
            logger.info(f"Added {len(ids)} documents with HNSW indexing")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def as_retriever(
        self, 
        k: int = 3, 
        source_normalized: Optional[str] = None,
        ranker_type: str = "weighted",
        ranker_weights: List[float] = None,
        mmr: bool = True,
        fetch_k: int = 12
    ) -> BaseRetriever:
        """Create retriever with hybrid search and optional source filter."""
        ranker_weights = ranker_weights or [0.6, 0.4]
        
        search_kwargs = {
            "k": k,
            "mmr": mmr,
            "fetch_k": fetch_k,
            "param": {
                "ef": max(k*4, 128)  # HNSW search parameter [(2)](https://milvus.io/docs/hnsw-sq.md#Index-params)
            }
        }
        
        # Add source filter if provided
        if source_normalized:
            search_kwargs["expr"] = f'source_normalized == "{source_normalized}"'
            logger.info(f"Applied source filter: {source_normalized}")
        
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs,
            ranker_type=ranker_type,
            ranker_params={"weights": ranker_weights}
        )
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 4, 
        source_normalized: str = None
    ) -> List[Document]:
        """Perform hybrid search with optional source filtering."""
        search_params = {
            "param": {
                "ef": max(k * 4, 128)  # Increased ef for better recall 
            }
        }
        
        # Log collection size
        collection = Collection(self.collection_name)
        collection_stats = collection.num_entities
        logger.info(f"Collection size: {collection_stats} documents")
        
        expr = f'source_normalized == "{source_normalized}"' if source_normalized else None
        logger.info(f"Search query: '{query}' with k={k}")
        logger.info(f"Search params: {search_params}")
        logger.info(f"Filter expression: {expr}")
        
        try:
            results = self.vector_store.similarity_search(
                query, 
                k=k, 
                expr=expr,
                search_params=search_params
            )
            
            logger.info(f"Initial search returned {len(results)} results")
            
            # Detailed results logging
            if results:
                for i, doc in enumerate(results):
                    metadata = getattr(doc, 'metadata', {})
                    source = metadata.get('source_normalized', 'N/A')
                    score = metadata.get('score', 'N/A')
                    logger.info(f"Result {i+1}: source={source}, score={score}")
            
            # Fallback without filter if no results and filter was applied
            if not results and source_normalized:
                logger.info(f"No results with filter '{expr}', trying without filter")
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    search_params=search_params
                )
                logger.info(f"Fallback search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return []