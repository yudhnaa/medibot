#!/usr/bin/env python3
"""Find disease titles related to query diseases using OpenRouter embeddings."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import requests

DEFAULT_INPUT = "processed_data/merged_final/merged_final.csv"
DEFAULT_OUTPUT = "outputs/related_diseases.csv"
DEFAULT_LUNG_OUTPUT = "processed_data/lung_dataset.csv"
DEFAULT_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_QUERIES = ("COVID-19", "Lung")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
LOGGER = logging.getLogger("find_related_diseases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find disease titles related to COVID-19/Lung using embeddings."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--lung-output",
        default=DEFAULT_LUNG_OUTPUT,
        help="CSV path for records above threshold",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Only copy records above threshold to lung output from existing output CSV",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=list(DEFAULT_QUERIES),
        help="Disease/query titles to compare against",
    )
    parser.add_argument("--title-column", default="title", help="CSV title column name")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="OpenRouter embedding model"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Minimum cosine similarity to copy into lung output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of titles per embedding request",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between embedding requests",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing OpenRouter API key",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level",
    )
    return parser.parse_args()


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_records(
    csv_path: Path, title_column: str
) -> tuple[list[dict[str, str]], list[str]]:
    LOGGER.info(
        "Reading input CSV", extra={"path": str(csv_path), "title_column": title_column}
    )
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or title_column not in reader.fieldnames:
            columns = ", ".join(reader.fieldnames or [])
            raise ValueError(f"Column {title_column!r} not found. Available: {columns}")
        records = [row for row in reader if row.get(title_column, "").strip()]
        fieldnames = list(reader.fieldnames)
    LOGGER.info("Loaded records", extra={"count": len(records)})
    return records, fieldnames


def write_records(
    csv_path: Path, fieldnames: list[str], records: list[dict[str, str]]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Writing records CSV", extra={"path": str(csv_path), "rows": len(records)}
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_scores(csv_path: Path) -> dict[str, float]:
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not {"original_title", "similarity_score"}.issubset(
            reader.fieldnames
        ):
            columns = ", ".join(reader.fieldnames or [])
            raise ValueError(
                "Output CSV must contain original_title and similarity_score. "
                f"Available: {columns}"
            )
        return {
            row["original_title"].strip(): float(row["similarity_score"])
            for row in reader
            if row.get("original_title", "").strip()
        }


def embed_texts(texts: list[str], *, api_key: str, model: str) -> np.ndarray:
    LOGGER.debug("Requesting embeddings", extra={"count": len(texts), "model": model})
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException:
        LOGGER.exception(
            "Embedding request failed", extra={"count": len(texts), "model": model}
        )
        raise

    payload = response.json()
    embeddings = [item["embedding"] for item in payload["data"]]
    LOGGER.debug(
        "Received embeddings", extra={"count": len(embeddings), "model": model}
    )
    return np.asarray(embeddings, dtype=np.float32)


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def copy_matched_records(
    records: list[dict[str, str]],
    fieldnames: list[str],
    output_path: Path,
    lung_output_path: Path,
    title_column: str,
    threshold: float,
) -> int:
    scores_by_title = load_scores(output_path)
    matched_records = [
        record
        for record in records
        if scores_by_title.get(record[title_column].strip(), 0.0) > threshold
    ]
    write_records(lung_output_path, fieldnames, matched_records)
    return len(matched_records)


def calculate_scores(
    titles: list[str],
    queries: list[str],
    *,
    api_key: str,
    model: str,
    batch_size: int,
    sleep_seconds: float,
    threshold: float,
    records_by_title: dict[str, dict[str, str]],
) -> tuple[list[tuple[str, float]], list[dict[str, str]]]:
    LOGGER.info("Embedding query titles", extra={"query_count": len(queries)})
    query_embeddings = normalize(embed_texts(queries, api_key=api_key, model=model))

    results: list[tuple[str, float]] = []
    matched_records: list[dict[str, str]] = []
    total_batches = (len(titles) + batch_size - 1) // batch_size
    for batch_number, batch in enumerate(batched(titles, batch_size), start=1):
        LOGGER.info(
            "Processing title batch",
            extra={
                "batch": batch_number,
                "total_batches": total_batches,
                "batch_size": len(batch),
            },
        )
        title_embeddings = normalize(embed_texts(batch, api_key=api_key, model=model))
        scores = title_embeddings @ query_embeddings.T
        max_scores = scores.max(axis=1)
        matched_count = 0
        for title, score in zip(batch, max_scores, strict=True):
            score_value = float(score)
            results.append((title, score_value))
            if score_value > threshold:
                matched_count += 1
                matched_records.append(records_by_title[title])
        LOGGER.info(
            "Finished title batch",
            extra={
                "batch": batch_number,
                "total_batches": total_batches,
                "matched_count": matched_count,
                "total_matched": len(results),
            },
        )
        if sleep_seconds > 0:
            LOGGER.debug("Sleeping between batches", extra={"seconds": sleep_seconds})
            time.sleep(sleep_seconds)
    return results, matched_records


def write_score_records(output_path: Path, results: list[tuple[str, float]]) -> None:
    results.sort(key=lambda item: item[1], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Writing output CSV", extra={"path": str(output_path), "rows": len(results)}
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["original_title", "similarity_score"]
        )
        writer.writeheader()
        for title, score in results:
            writer.writerow(
                {"original_title": title, "similarity_score": f"{score:.6f}"}
            )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    LOGGER.info(
        "Starting disease similarity export",
        extra={
            "input": args.input,
            "output": args.output,
            "queries": args.queries,
            "model": args.model,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
        },
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    lung_output_path = Path(args.lung_output)
    try:
        records, fieldnames = read_records(input_path, args.title_column)
        if not records:
            LOGGER.error("No records found", extra={"path": str(input_path)})
            print("No records found.", file=sys.stderr)
            return 1

        if args.copy_only:
            matched_count = copy_matched_records(
                records,
                fieldnames,
                output_path,
                lung_output_path,
                args.title_column,
                args.threshold,
            )
            print(f"Wrote {matched_count} rows to {lung_output_path}")
            return 0

        api_key = os.getenv(args.api_key_env)
        if not api_key:
            LOGGER.error("Missing API key", extra={"env_var": args.api_key_env})
            print(
                f"Missing {args.api_key_env}. Export OpenRouter API key first.",
                file=sys.stderr,
            )
            return 2

        titles = [record[args.title_column].strip() for record in records]
        records_by_title = {
            record[args.title_column].strip(): record for record in records
        }

        results, matched_records = calculate_scores(
            titles,
            args.queries,
            api_key=api_key,
            model=args.model,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep,
            threshold=args.threshold,
            records_by_title=records_by_title,
        )
        write_score_records(output_path, results)
        write_records(lung_output_path, fieldnames, matched_records)
    except Exception:
        LOGGER.exception("Disease similarity export failed")
        raise

    LOGGER.info(
        "Disease similarity export completed",
        extra={
            "rows": len(results),
            "output": str(output_path),
            "lung_rows": len(matched_records),
            "lung_output": str(lung_output_path),
        },
    )
    print(f"Wrote {len(results)} rows to {output_path}")
    print(f"Wrote {len(matched_records)} rows to {lung_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
