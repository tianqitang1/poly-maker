"""
Semantic Search Engine for Poly-Maker

Provides semantic similarity search for news and market matching using:
- Local embeddings (sentence-transformers) - Free, fast
- Cloud embeddings (OpenAI, Cohere, Gemini) - Higher quality

Features:
- ChromaDB vector storage
- Market embedding cache with automatic cleanup
- Multiple embedding provider support
- Batch processing for efficiency
- Similarity threshold filtering

Usage:
    from poly_utils.semantic_search import SemanticSearchEngine

    config = {
        'enabled': True,
        'provider': 'sentence_transformer',  # or 'openai', 'cohere', 'gemini'
        'model': 'all-MiniLM-L6-v2',
        'similarity_threshold': 0.5,
        'cache_markets': True
    }

    engine = SemanticSearchEngine(config)

    # Add news items
    engine.add_news(news_items)

    # Cache market embeddings
    engine.cache_market("Will Bitcoin reach $100k?", market_id="btc-100k")

    # Search for similar news
    results = engine.search_news("Will Bitcoin reach $100k?", max_results=5)
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from poly_utils.logging_utils import get_logger

logger = get_logger('poly_utils.semantic_search')


@dataclass
class SearchResult:
    """Result from semantic search."""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float
    id: str


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.total_embeddings = 0
        self.total_cost = 0.0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_embeddings': self.total_embeddings,
            'total_cost': self.total_cost
        }


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers embeddings (free, fast)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model', 'all-MiniLM-L6-v2')

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using sentence-transformers."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.total_embeddings += len(texts)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        # Model-specific dimensions
        dimensions = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'multi-qa-MiniLM-L6-cos-v1': 384,
        }
        return dimensions.get(self.model_name, 384)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings (cloud, high quality)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('model', 'text-embedding-3-small')
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI embeddings: {self.model}")
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI."""
        # Batch up to 2048 texts per request (OpenAI limit)
        all_embeddings = []

        for i in range(0, len(texts), 2048):
            batch = texts[i:i + 2048]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )

            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

            # Track cost (text-embedding-3-small: $0.02 per 1M tokens, ~750 tokens per text)
            estimated_tokens = len(batch) * 750
            cost = (estimated_tokens / 1_000_000) * 0.02
            self.total_cost += cost

        self.total_embeddings += len(texts)
        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
        }
        return dimensions.get(self.model, 1536)


class CohereEmbedder(BaseEmbedder):
    """Cohere embeddings (cloud, optimized for search)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('model', 'embed-english-v3.0')
        api_key_env = config.get('api_key_env', 'COHERE_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        try:
            import cohere
            self.client = cohere.Client(self.api_key)
            logger.info(f"Initialized Cohere embeddings: {self.model}")
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Cohere."""
        # Cohere supports up to 96 texts per request
        all_embeddings = []

        for i in range(0, len(texts), 96):
            batch = texts[i:i + 96]

            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type='search_document'  # Optimized for storage
            )

            all_embeddings.extend(response.embeddings)

            # Track cost (embed-english-v3.0: $0.10 per 1M tokens)
            estimated_tokens = len(batch) * 500
            cost = (estimated_tokens / 1_000_000) * 0.10
            self.total_cost += cost

        self.total_embeddings += len(texts)
        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        dimensions = {
            'embed-english-v3.0': 1024,
            'embed-english-light-v3.0': 384,
            'embed-multilingual-v3.0': 1024,
        }
        return dimensions.get(self.model, 1024)


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embeddings (cloud, integrated with Gemini API)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get('model', 'models/embedding-001')
        api_key_env = config.get('api_key_env', 'GEMINI_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Missing API key: {api_key_env} not set in environment")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            logger.info(f"Initialized Gemini embeddings: {self.model}")
        except ImportError:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini."""
        all_embeddings = []

        # Gemini supports batch embedding
        for text in texts:
            result = self.genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            all_embeddings.append(result['embedding'])

            # Very cheap - roughly $0.00001 per 1000 characters
            cost = (len(text) / 1000) * 0.00001
            self.total_cost += cost

        self.total_embeddings += len(texts)
        return all_embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return 768  # Gemini embedding-001


class MarketEmbeddingCache:
    """
    Cache for market embeddings with automatic cleanup.

    Stores market embeddings to avoid re-computing them on every search.
    Automatically removes closed/expired markets.
    """

    def __init__(self, cache_dir: str = ".cache/market_embeddings"):
        self.cache_dir = cache_dir
        self.cache = {}  # market_id -> {embedding, question, timestamp, metadata}

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load existing cache
        self._load_cache()

    def add(
        self,
        market_id: str,
        question: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add market embedding to cache."""
        self.cache[market_id] = {
            'question': question,
            'embedding': embedding,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        logger.debug(f"Cached embedding for market: {market_id}")

    def get(self, market_id: str) -> Optional[List[float]]:
        """Get cached embedding for a market."""
        if market_id in self.cache:
            return self.cache[market_id]['embedding']
        return None

    def remove(self, market_id: str):
        """Remove market from cache."""
        if market_id in self.cache:
            del self.cache[market_id]
            logger.debug(f"Removed market from cache: {market_id}")

    def cleanup_closed_markets(self, closed_market_ids: List[str]):
        """Remove closed markets from cache."""
        for market_id in closed_market_ids:
            self.remove(market_id)

        logger.info(f"Cleaned up {len(closed_market_ids)} closed markets from cache")

    def cleanup_old_markets(self, max_age_hours: int = 168):
        """Remove markets older than max_age_hours (default: 7 days)."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        old_markets = [
            market_id for market_id, data in self.cache.items()
            if data['timestamp'] < cutoff_time
        ]

        for market_id in old_markets:
            self.remove(market_id)

        if old_markets:
            logger.info(f"Cleaned up {len(old_markets)} old markets from cache")

    def save(self):
        """Save cache to disk."""
        cache_file = os.path.join(self.cache_dir, 'market_cache.json')

        with open(cache_file, 'w') as f:
            json.dump(self.cache, f)

        logger.debug(f"Saved market embedding cache ({len(self.cache)} markets)")

    def _load_cache(self):
        """Load cache from disk."""
        cache_file = os.path.join(self.cache_dir, 'market_cache.json')

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded market embedding cache ({len(self.cache)} markets)")
            except Exception as e:
                logger.warning(f"Failed to load market cache: {e}")
                self.cache = {}


class SemanticSearchEngine:
    """
    Semantic search engine using ChromaDB and multiple embedding providers.

    Supports:
    - News item embeddings and search
    - Market embedding cache with cleanup
    - Multiple embedding providers (local + cloud)
    - Similarity threshold filtering
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize semantic search engine.

        Args:
            config: Configuration dict with embedding provider and ChromaDB settings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            )

        self.config = config
        self.enabled = config.get('enabled', True)
        self.provider_name = config.get('provider', 'sentence_transformer').lower()
        self.similarity_threshold = config.get('similarity_threshold', 0.5)
        self.cache_markets = config.get('cache_markets', True)

        # Initialize embedding provider
        self.embedder = self._init_embedder()

        # Initialize ChromaDB
        chroma_dir = config.get('chroma_dir', '.cache/chromadb')
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)

        # Create collections
        embedding_dim = self.embedder.get_dimension()
        self.news_collection = self.chroma_client.get_or_create_collection(
            name="news_items",
            metadata={"dimension": embedding_dim}
        )
        self.market_collection = self.chroma_client.get_or_create_collection(
            name="market_cache",
            metadata={"dimension": embedding_dim}
        )

        # Market embedding cache
        self.market_cache = MarketEmbeddingCache() if self.cache_markets else None

        logger.info(
            f"Initialized SemanticSearchEngine (provider={self.provider_name}, "
            f"dim={embedding_dim}, cache_markets={self.cache_markets})"
        )

    def _init_embedder(self) -> BaseEmbedder:
        """Initialize embedding provider based on config."""
        if self.provider_name == 'sentence_transformer':
            return SentenceTransformerEmbedder(self.config)
        elif self.provider_name == 'openai':
            return OpenAIEmbedder(self.config)
        elif self.provider_name == 'cohere':
            return CohereEmbedder(self.config)
        elif self.provider_name == 'gemini':
            return GeminiEmbedder(self.config)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider_name}")

    def add_news(self, news_items: List[Any], batch_size: int = 100):
        """
        Add news items to the semantic search index.

        Args:
            news_items: List of NewsItem objects
            batch_size: Number of items to process in each batch
        """
        if not news_items:
            return

        # Process in batches
        for i in range(0, len(news_items), batch_size):
            batch = news_items[i:i + batch_size]

            # Prepare texts and metadata
            texts = []
            metadatas = []
            ids = []

            for item in batch:
                # Combine title and summary for embedding
                text = f"{item.title} {item.summary}"
                texts.append(text)

                # Store metadata
                metadatas.append(item.to_dict())

                # Use link as unique ID
                ids.append(item.link)

            # Embed texts
            embeddings = self.embedder.embed(texts)

            # Add to ChromaDB
            try:
                self.news_collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                # Handle duplicates - update instead
                logger.debug(f"Some news items already exist, skipping: {e}")

        logger.info(f"Added {len(news_items)} news items to semantic search index")

    def cache_market(
        self,
        question: str,
        market_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache market embedding for faster future searches.

        Args:
            question: Market question text
            market_id: Unique market identifier
            metadata: Optional market metadata (price, volume, etc.)
        """
        if not self.cache_markets:
            return

        # Check if already cached
        if self.market_cache.get(market_id):
            return

        # Embed question
        embedding = self.embedder.embed([question])[0]

        # Add to cache
        self.market_cache.add(market_id, question, embedding, metadata)

        # Also add to ChromaDB for persistence
        try:
            self.market_collection.add(
                embeddings=[embedding],
                documents=[question],
                metadatas=[metadata or {}],
                ids=[market_id]
            )
        except Exception as e:
            logger.debug(f"Market already in ChromaDB: {e}")

    def search_news(
        self,
        query: str,
        market_id: Optional[str] = None,
        max_results: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for news items semantically similar to query.

        Args:
            query: Query text (e.g., market question)
            market_id: Optional market ID to use cached embedding
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold (overrides config)

        Returns:
            List of SearchResult objects
        """
        # Use cached market embedding if available
        if market_id and self.market_cache:
            query_embedding = self.market_cache.get(market_id)
            if not query_embedding:
                # Cache miss - embed and cache
                query_embedding = self.embedder.embed([query])[0]
                self.cache_market(query, market_id)
        else:
            # No cache - embed query
            query_embedding = self.embedder.embed([query])[0]

        # Query ChromaDB
        results = self.news_collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results * 2  # Get more results for filtering
        )

        # Parse results
        search_results = []
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold

        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity

                # Filter by threshold
                if similarity < threshold:
                    continue

                search_results.append(SearchResult(
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    similarity_score=similarity,
                    id=results['ids'][0][i]
                ))

        # Sort by similarity and limit
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results[:max_results]

    def cleanup_closed_markets(self, closed_market_ids: List[str]):
        """Remove closed markets from cache and ChromaDB."""
        if self.market_cache:
            self.market_cache.cleanup_closed_markets(closed_market_ids)

        # Remove from ChromaDB
        try:
            self.market_collection.delete(ids=closed_market_ids)
            logger.info(f"Removed {len(closed_market_ids)} closed markets from ChromaDB")
        except Exception as e:
            logger.warning(f"Failed to remove closed markets from ChromaDB: {e}")

    def cleanup_old_markets(self, max_age_hours: int = 168):
        """Remove markets older than max_age_hours from cache."""
        if self.market_cache:
            self.market_cache.cleanup_old_markets(max_age_hours)

    def clear_news(self):
        """Clear all news items from the index."""
        self.chroma_client.delete_collection("news_items")
        self.news_collection = self.chroma_client.create_collection("news_items")
        logger.info("Cleared news items from semantic search index")

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'provider': self.provider_name,
            'embedding_stats': self.embedder.get_stats(),
            'news_count': self.news_collection.count(),
            'cached_markets': len(self.market_cache.cache) if self.market_cache else 0,
            'similarity_threshold': self.similarity_threshold
        }

    def save(self):
        """Save caches to disk."""
        if self.market_cache:
            self.market_cache.save()
