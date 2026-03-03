from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .settings import settings

@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str

def _demo_docs() -> List[Doc]:
    # Demo corpus: keep it small but realistic
    return [
        Doc("d1", "Value at Risk (VaR) estimates the maximum expected loss over a time horizon at a given confidence level."),
        Doc("d2", "Conditional VaR (CVaR), also called Expected Shortfall, is the average loss in the worst alpha% of cases."),
        Doc("d3", "A bi-encoder embeds queries and documents separately, enabling fast nearest-neighbor search using vector indexes."),
        Doc("d4", "A cross-encoder jointly encodes a query-document pair to produce a relevance score, typically improving ranking."),
        Doc("d5", "FAISS is a library for efficient similarity search and clustering of dense vectors, often used for ANN retrieval."),
        Doc("d6", "Sharpe ratio measures risk-adjusted return: mean excess return divided by return standard deviation."),
        Doc("d7", "Maximum drawdown is the largest peak-to-trough decline in portfolio value over a specified period."),
        Doc("d8", "Portfolio optimization often balances expected return and risk, such as variance, under constraints on weights."),
        Doc("d9", "In retrieval QA, you retrieve candidate passages first, then select or generate an answer conditioned on them."),
        Doc("d10", "Sentence Transformers provide pre-trained models to produce semantically meaningful embeddings for text."),
    ]

def build_demo_index() -> Tuple[List[Doc], faiss.Index, SentenceTransformer]:
    """
    Build a FAISS index over demo documents using a bi-encoder.
    Returns (docs, index, embedder).
    """
    docs = _demo_docs()
    embedder = SentenceTransformer(settings.bi_encoder_model)

    texts = [d.text for d in docs]
    embs = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=settings.normalize_embeddings,
        show_progress_bar=False,
    ).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # dot-product; with normalized vectors => cosine similarity
    index.add(embs)
    return docs, index, embedder
