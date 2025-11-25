"""
Baseline retrieval methods for comparison.
"""

from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
import math
from .vector_index import VectorIndex
from .embedder import Embedder


class BM25Retriever:
    """
    BM25 (Best Matching 25) keyword-based retrieval baseline.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.avg_doc_length = 0.0
    
    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Document identifier
            text: Document text
        """
        tokens = self._tokenize(text)
        doc_length = len(tokens)
        term_freq = Counter(tokens)
        
        self.documents.append({
            'id': doc_id,
            'text': text,
            'tokens': tokens,
            'length': doc_length,
            'term_freq': term_freq
        })
        
        # Update document frequencies
        for term in set(tokens):
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def build_index(self) -> None:
        """
        Build the BM25 index (compute IDF values).
        """
        num_docs = len(self.documents)
        if num_docs == 0:
            return
        
        # Compute IDF
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        # Compute average document length
        total_length = sum(doc['length'] for doc in self.documents)
        self.avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search documents using BM25.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc in self.documents:
            score = 0.0
            doc_length = doc['length']
            term_freq = doc['term_freq']
            
            for term in query_tokens:
                if term in term_freq:
                    tf = term_freq[term]
                    idf = self.idf.get(term, 0.0)
                    
                    # BM25 formula
                    numerator = idf * tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    score += numerator / denominator
            
            scores.append((doc['id'], score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


class HybridRetriever:
    """
    Hybrid retrieval combining vector and keyword search.
    """
    
    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_retriever: BM25Retriever,
        embedder: Embedder,
        alpha: float = 0.7
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_index: Vector similarity index
            bm25_retriever: BM25 retriever
            embedder: Embedder for query encoding
            alpha: Weight for vector search (1-alpha for BM25)
        """
        self.vector_index = vector_index
        self.bm25_retriever = bm25_retriever
        self.embedder = embedder
        self.alpha = alpha
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Hybrid search combining vector and BM25.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (node_id, score) tuples
        """
        # Vector search
        query_emb = self.embedder.embed(query)
        vector_results = self.vector_index.search(query_emb, candidates=None, k=k * 2)
        vector_scores = {node_id: score for node_id, score in vector_results}
        
        # BM25 search
        bm25_results = self.bm25_retriever.search(query, k=k * 2)
        bm25_scores = {node_id: score for node_id, score in bm25_results}
        
        # Normalize scores
        all_nodes = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        if len(all_nodes) == 0:
            return []
        
        # Normalize vector scores
        if vector_scores:
            max_vec = max(vector_scores.values())
            min_vec = min(vector_scores.values())
            vec_range = max_vec - min_vec if max_vec != min_vec else 1.0
            vector_scores = {nid: (score - min_vec) / vec_range 
                           for nid, score in vector_scores.items()}
        
        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
            bm25_scores = {nid: (score - min_bm25) / bm25_range 
                          for nid, score in bm25_scores.items()}
        
        # Combine scores
        combined_scores = {}
        for node_id in all_nodes:
            vec_score = vector_scores.get(node_id, 0.0)
            bm25_score = bm25_scores.get(node_id, 0.0)
            combined_scores[node_id] = self.alpha * vec_score + (1 - self.alpha) * bm25_score
        
        # Sort and return top k
        results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:k]

