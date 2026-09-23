#!/usr/bin/env python3
"""
Hybrid retrieval engine: FAISS dense + BM25 sparse + RRF fusion + cross-encoder reranking.

Architecture:
  1. Dense retrieval  — FAISS IndexFlatIP with BAAI/bge-large-en-v1.5 embeddings (cosine)
  2. Sparse retrieval — BM25Okapi via rank-bm25 on whitespace-tokenised text
  3. RRF fusion       — Reciprocal Rank Fusion merges both ranked lists
  4. Reranking        — BAAI/bge-reranker-large cross-encoder scores top-K candidates

Usage:
    from src.retriever import HybridRetriever

    retriever = HybridRetriever()
    retriever.build_from_ingest("data/filings/index.json")
    retriever.save("data/retriever_index")

    chunks = retriever.retrieve("What are Apple's supply chain risks?", top_k=5)
"""

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_CHUNKS", "20"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
RRF_K = 60  # standard constant from the original RRF paper (Cormack et al. 2009)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with provenance metadata and retrieval score."""
    chunk_id: str
    ticker: str
    form_type: str
    period: str
    section_name: str
    text: str
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


# ── Main class ────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Three-stage retrieval pipeline.

    Build once, retrieve many times. Supports optional ticker/filing-type
    pre-filtering before retrieval so queries stay scoped to relevant filings.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.embed_model_name = embed_model
        self.reranker_model_name = reranker_model

        self._chunks: List[Dict[str, Any]] = []
        self._faiss_index = None
        self._bm25 = None
        self._embedder = None
        self._reranker = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_from_ingest(self, index_path: str = "data/filings/index.json") -> None:
        """Load chunks from the ingest pipeline output and build both indexes."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Ingest index not found at '{index_path}'. "
                "Run: python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self._chunks = data["chunks"]
        print(f"[Retriever] Loaded {len(self._chunks)} chunks from {index_path}")
        self._build_dense_index()
        self._build_sparse_index()

    def build_from_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Build indexes directly from a list of chunk dicts (useful for testing)."""
        self._chunks = chunks
        self._build_dense_index()
        self._build_sparse_index()

    def retrieve(
        self,
        query: str,
        top_k: int = RERANKER_TOP_K,
        tickers: Optional[List[str]] = None,
        filing_types: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Run the full hybrid retrieval pipeline.

        Args:
            query: Natural language search query.
            top_k: Number of chunks to return after reranking.
            tickers: Restrict retrieval to these tickers (None = all).
            filing_types: Restrict retrieval to these form types (None = all).

        Returns:
            List of RetrievedChunk objects sorted by reranker score (highest first).
        """
        if not self._chunks:
            raise RuntimeError(
                "Index not built. Call build_from_ingest() or build_from_chunks() first."
            )

        candidate_indices = self._filter_indices(tickers, filing_types)
        if not candidate_indices:
            return []

        pool_size = min(TOP_K_RETRIEVAL * 2, len(candidate_indices))

        # Stage 1 & 2: dense + sparse retrieval
        dense_results = self._dense_retrieve(query, candidate_indices, pool_size)
        sparse_results = self._sparse_retrieve(query, candidate_indices, pool_size)

        # Stage 3: RRF fusion → single ranked list
        fused = _rrf_merge(dense_results, sparse_results, pool_size)

        # Stage 4: cross-encoder reranking
        return self._rerank(query, fused, top_k)

    def save(self, directory: str = "data/retriever_index") -> None:
        """Persist the FAISS index, BM25 model, and chunks to disk."""
        import faiss

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(path / "faiss.index"))
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, indent=2)

        print(f"[Retriever] Index saved to {directory}/")

    def load(self, directory: str = "data/retriever_index") -> None:
        """Load a previously saved retriever index from disk."""
        import faiss

        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Retriever index directory not found: {directory}")

        self._faiss_index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "bm25.pkl", "rb") as f:
            self._bm25 = pickle.load(f)
        with open(path / "chunks.json", encoding="utf-8") as f:
            self._chunks = json.load(f)

        print(f"[Retriever] Loaded {len(self._chunks)} chunks from {directory}/")

    # ── Dense retrieval (FAISS + BGE embeddings) ───────────────────────────────

    def _build_dense_index(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"[Retriever] Loading embedding model: {self.embed_model_name}")
        self._embedder = SentenceTransformer(self.embed_model_name)

        texts = [c["text"] for c in self._chunks]
        print(f"[Retriever] Embedding {len(texts)} chunks (this may take a few minutes)…")
        embeddings = self._embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # normalise → cosine similarity via inner product
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        # IndexFlatIP = exact inner product search (cosine on normalised vectors)
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(embeddings)
        print(f"[Retriever] FAISS index built: {self._faiss_index.ntotal} vectors, dim={dim}")

    def _dense_retrieve(
        self,
        query: str,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Return (global_chunk_idx, cosine_score) pairs for the top-k dense results."""
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Over-retrieve from FAISS then filter to candidate set
        search_k = min(top_k * 4, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(q_emb, search_k)

        candidate_set = set(candidate_indices)
        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if int(idx) in candidate_set
        ]
        return results[:top_k]

    # ── Sparse retrieval (BM25) ────────────────────────────────────────────────

    def _build_sparse_index(self) -> None:
        from rank_bm25 import BM25Okapi

        tokenised = [_tokenise(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenised)
        print("[Retriever] BM25 index built.")

    def _sparse_retrieve(
        self,
        query: str,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """Return (global_chunk_idx, bm25_score) pairs for the top-k sparse results."""
        query_tokens = _tokenise(query)
        all_scores = self._bm25.get_scores(query_tokens)

        # Mask to candidate set, sort, and return top-k
        masked = [(i, float(all_scores[i])) for i in candidate_indices]
        masked.sort(key=lambda x: x[1], reverse=True)
        return masked[:top_k]

    # ── Cross-encoder reranking ────────────────────────────────────────────────

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Re-score RRF candidates with a cross-encoder and return top_k RetrievedChunks."""
        from sentence_transformers import CrossEncoder

        if not candidates:
            return []

        if self._reranker is None:
            print(f"[Retriever] Loading reranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)

        pairs = [(query, self._chunks[idx]["text"]) for idx, _ in candidates]
        rerank_scores = self._reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]

        results = []
        for (idx, _), score in ranked:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    ticker=chunk["ticker"],
                    form_type=chunk["form_type"],
                    period=chunk["period"],
                    section_name=chunk.get("section_name", chunk.get("section_key", "")),
                    text=chunk["text"],
                    score=float(score),
                )
            )
        return results

    # ── Candidate filtering ────────────────────────────────────────────────────

    def _filter_indices(
        self,
        tickers: Optional[List[str]],
        filing_types: Optional[List[str]],
    ) -> List[int]:
        """Return indices of chunks matching the optional ticker / filing-type filters."""
        indices = []
        for i, chunk in enumerate(self._chunks):
            if tickers and chunk.get("ticker") not in tickers:
                continue
            if filing_types and chunk.get("form_type") not in filing_types:
                continue
            indices.append(i)
        # If nothing matches filters, fall back to full corpus
        return indices if indices else list(range(len(self._chunks)))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Simple whitespace tokeniser for BM25 (lowercase)."""
    return text.lower().split()


def _rrf_merge(
    dense: List[Tuple[int, float]],
    sparse: List[Tuple[int, float]],
    top_k: int,
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion.
    score(doc) = Σ_i  1 / (k + rank_i(doc))
    Returns list of (global_chunk_idx, rrf_score) sorted descending.
    """
    scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(sparse):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:top_k]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the retriever index.")
    parser.add_argument("--build", action="store_true", help="Build index from ingest output")
    parser.add_argument("--query", type=str, help="Test query to run after building")
    parser.add_argument(
        "--index-path", default="data/filings/index.json", help="Path to ingest index.json"
    )
    parser.add_argument(
        "--save-dir", default="data/retriever_index", help="Directory to save/load index"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    retriever = HybridRetriever()

    if args.build:
        retriever.build_from_ingest(args.index_path)
        retriever.save(args.save_dir)
    else:
        retriever.load(args.save_dir)

    if args.query:
        print(f"\nQuery: {args.query}\n{'='*60}")
        results = retriever.retrieve(args.query, top_k=args.top_k)
        for i, chunk in enumerate(results, 1):
            print(f"\n[{i}] score={chunk.score:.4f}  {chunk.ticker} | {chunk.form_type} | {chunk.period} | {chunk.section_name}")
            print(chunk.text[:300] + "…")
