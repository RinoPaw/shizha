"""Rebuild the local semantic search index after the dataset changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anti_fraud_explorer import config
from anti_fraud_explorer.dataset import FraudCase, load_dataset
from anti_fraud_explorer.embeddings import (
    EmbeddingClient,
    build_embedding_text,
    normalize_vector,
    write_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or rebuild data/embeddings/case_embeddings.json. "
            "Use --no-resume after data/processed/case_items.json changes."
        )
    )
    parser.add_argument("--output", type=Path, default=config.EMBEDDING_INDEX_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Only index the first N items.")
    parser.add_argument("--batch-size", type=int, default=config.EMBEDDING_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.EMBEDDING_WORKERS)
    parser.add_argument("--request-timeout", type=float, default=config.EMBEDDING_REQUEST_TIMEOUT)
    parser.add_argument("--retry-delay", type=float, default=config.EMBEDDING_REQUEST_DELAY)
    parser.add_argument("--delay", dest="retry_delay", type=float)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not config.EMBEDDING_API_KEY:
        raise SystemExit("EMBEDDING_API_KEY is not configured. Add it to .env first.")

    kb = load_dataset()
    items = kb.items[: args.limit] if args.limit > 0 else kb.items
    payload = initial_payload(kb)
    indexed = {}

    if args.output.exists() and not args.no_resume:
        payload = load_existing_payload(args.output, kb)
        indexed = {
            str(row["id"]): row
            for row in payload.get("embeddings", [])
            if row.get("id") and row.get("embedding")
        }

    payload["embeddings"] = [indexed[item.id] for item in items if item.id in indexed]
    write_index(payload, args.output)

    print(
        "Building embedding index "
        f"for {len(items)} items with model {config.EMBEDDING_MODEL} "
        f"({indexed_item_count(items, indexed)} already indexed, workers={args.workers}, "
        f"timeout={args.request_timeout}s)...",
        flush=True,
    )

    for round_no in range(1, args.max_rounds + 1):
        remaining = [item for item in items if item.id not in indexed]
        if not remaining:
            print(f"Embedding index is complete: {args.output}", flush=True)
            return

        batches = list(chunked(remaining, args.batch_size))
        print(
            f"Round {round_no}: {len(remaining)} item(s) left, "
            f"{len(batches)} batch(es) queued.",
            flush=True,
        )
        failed_batches = run_round(args, batches, indexed, payload, items)
        if not failed_batches:
            continue

        print(
            f"Round {round_no}: {len(failed_batches)} batch(es) failed; "
            f"{progress_text(indexed_item_count(items, indexed), len(items))}.",
            flush=True,
        )
        if args.retry_delay > 0:
            time.sleep(args.retry_delay)

    missing = len([item for item in items if item.id not in indexed])
    raise SystemExit(f"Embedding index incomplete after {args.max_rounds} rounds: {missing} missing.")


def run_round(
    args: argparse.Namespace,
    batches: list[list[FraudCase]],
    indexed: dict[str, dict[str, Any]],
    payload: dict[str, Any],
    items: list[FraudCase],
) -> list[list[FraudCase]]:
    failed_batches = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(embed_batch, batch, args.request_timeout): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                rows = future.result()
            except Exception as exc:  # noqa: BLE001 - failed batches are retried later.
                print(
                    "Failed batch "
                    f"{batch[0].id}..{batch[-1].id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                failed_batches.append(batch)
                continue

            indexed.update({row["id"]: row for row in rows})
            payload["embeddings"] = [indexed[item.id] for item in items if item.id in indexed]
            payload["dimensions"] = payload["dimensions"] or len(rows[0]["embedding"])
            payload["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            write_index(payload, args.output)
            print(f"Indexed {progress_text(len(payload['embeddings']), len(items))}", flush=True)

    return failed_batches


def initial_payload(kb) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "schema_version": kb.schema_version,
            "generated_at": kb.generated_at,
            "item_count": len(kb.items),
        },
        "base_url": config.EMBEDDING_BASE_URL.rstrip("/"),
        "model": config.EMBEDDING_MODEL,
        "dimensions": 0,
        "embeddings": [],
    }


def load_existing_payload(path: Path, kb) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("model") != config.EMBEDDING_MODEL:
        raise SystemExit("Existing embedding index uses another model. Use --no-resume.")
    dataset = payload.get("dataset") or {}
    if (
        dataset.get("schema_version") != kb.schema_version
        or dataset.get("generated_at") != kb.generated_at
        or dataset.get("item_count") != len(kb.items)
    ):
        raise SystemExit(
            "Existing embedding index was built for a different dataset. "
            "Run again with --no-resume to rebuild every item."
        )
    return payload


def embed_batch(batch: list[FraudCase], request_timeout: float) -> list[dict[str, Any]]:
    client = EmbeddingClient(timeout=request_timeout, max_retries=0)
    texts = [build_embedding_text(item) for item in batch]
    vectors = client.embed_texts(texts)
    if len(vectors) != len(batch):
        raise RuntimeError("Embedding API returned a mismatched batch length.")
    rows = []
    for item, vector in zip(batch, vectors, strict=True):
        rows.append({"id": item.id, "embedding": normalize_vector(vector)})
    return rows


def chunked(items: list[FraudCase], size: int):
    for start in range(0, len(items), max(size, 1)):
        yield items[start : start + size]


def progress_text(done: int, total: int) -> str:
    percent = (done / total * 100) if total else 100
    return f"{done}/{total} ({percent:.1f}%)"


def indexed_item_count(items: list[FraudCase], indexed: dict[str, dict[str, Any]]) -> int:
    return len([item for item in items if item.id in indexed])


if __name__ == "__main__":
    main()
