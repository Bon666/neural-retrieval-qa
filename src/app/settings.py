from __future__ import annotations
from pydantic import BaseModel

class Settings(BaseModel):
    bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    normalize_embeddings: bool = True
    demo_k: int = 50  # default retrieve candidates

settings = Settings()
