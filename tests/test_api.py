"""Integration tests for FastAPI endpoints."""

import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "RAGme API"


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_benchmarks():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "stage_A_chunking_sweep" in data
    assert "stage_C_retrieval_sweep" in data
    assert "stage_F_prompt_tournament" in data


def test_stats():
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["total_documents"] >= 6000
    assert data["total_chunks"] >= 60000


def test_search():
    response = client.post("/api/search", json={"query": "Central Motor Vehicles Rules", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["snippet"]


def test_feedback():
    response = client.post(
        "/api/feedback",
        json={
            "query": "Test query",
            "answer": "Test answer",
            "rating": "positive",
            "critique": "Accurate citations and great follow-up questions.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "recorded"
    assert data["feedback_id"].startswith("fb-")
