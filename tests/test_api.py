from __future__ import annotations
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["docs"] > 0

def test_ask_basic():
    r = client.post("/ask", json={"query": "What is VaR?", "k_retrieve": 10})
    assert r.status_code == 200
    j = r.json()
    assert j["query"] == "What is VaR?"
    assert "best" in j
    assert "debug_top" in j
    assert j["best"]["doc_id"] != ""
    assert isinstance(j["debug_top"], list)
    assert len(j["debug_top"]) <= 10
