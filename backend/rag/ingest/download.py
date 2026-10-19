"""Download the dataset's Parquet shards from Hugging Face into data/raw/.

Usage:
    uv run python -m rag.ingest.download
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

from rag.config import settings


def download() -> None:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=settings.hf_dataset,
        repo_type="dataset",
        allow_patterns=["data/*.parquet", "README.md", "manifest.json"],
        local_dir=settings.raw_dir,
    )
    shards = sorted((settings.raw_dir / "data").glob("*.parquet"))
    size_mb = sum(p.stat().st_size for p in shards) / 1e6
    print(f"Downloaded {len(shards)} shards ({size_mb:.0f} MB) to {path}")


if __name__ == "__main__":
    download()
