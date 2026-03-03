from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    k_retrieve: int = Field(default=50, ge=1, le=200)

class DebugCandidate(BaseModel):
    doc_id: str
    retriever_score: float
    reranker_score: Optional[float] = None

class BestAnswer(BaseModel):
    doc_id: str
    text: str
    retriever_score: float
    reranker_score: Optional[float] = None

class AskResponse(BaseModel):
    query: str
    best: BestAnswer
    debug_top: List[DebugCandidate]
