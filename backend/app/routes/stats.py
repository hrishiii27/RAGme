"""System statistics and benchmark leaderboard endpoint."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app.schemas import StatsResponse
from app.services import get_stats
from rag.pipeline import find_project_root

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def system_stats():
    return get_stats()


@router.get("/benchmarks")
async def benchmark_leaderboards():
    root = find_project_root()
    master_file = root / "reports" / "benchmark_master.json"
    if master_file.exists():
        return json.loads(master_file.read_text())
    return {"status": "pending", "message": "Master benchmark file not found."}


@router.get("/health")
async def health_check():
    return {"status": "ok"}
