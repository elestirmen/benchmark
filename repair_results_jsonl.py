#!/usr/bin/env python
"""Copy-first audit and deduplication for benchmark results.jsonl checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


KEY_FIELDS = ("direction", "query_variant", "search_mode", "model_id", "query_id")
CRITICAL_FIELDS = (
    *KEY_FIELDS,
    "model_sha256",
    "expected_center_easting_m",
    "expected_center_northing_m",
    "predicted_center_easting_m",
    "predicted_center_northing_m",
    "error_m",
    "error_px",
    "success_5m",
    "success_10m",
    "success_25m",
    "success_50m",
    "status",
    "reason",
    "normalization",
    "template_size_px",
    "source_query_raster",
    "source_map_raster",
)
OPERATIONAL_FIELDS = {
    "created_at_utc",
    "search_seconds",
    "query_inference_seconds",
    "map_build_seconds",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_digest(payload: Any) -> bytes:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def selected_payload(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def non_operational_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in OPERATIONAL_FIELDS}


def repair_results(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    source = run_dir / "results.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"results.jsonl bulunamadı: {source}")
    if (run_dir / ".benchmark_run.lock").exists():
        raise RuntimeError(
            f"Etkin run kilidi var; çalışan benchmark durdurulmadan onarım yapılmaz: {run_dir}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = source.with_name(f"results.pre_dedup_{stamp}.jsonl")
    temporary = source.with_name(f".results.dedup_{stamp}.jsonl.tmp")
    database = source.with_name(f".results.dedup_{stamp}.sqlite")
    report_path = run_dir / f"dedup_report_{stamp}.json"
    shutil.copy2(source, backup)

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE seen (
            direction TEXT NOT NULL,
            query_variant TEXT NOT NULL,
            search_mode TEXT NOT NULL,
            model_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            critical_hash BLOB NOT NULL,
            diagnostic_hash BLOB NOT NULL,
            PRIMARY KEY (direction, query_variant, search_mode, model_id, query_id)
        ) WITHOUT ROWID
        """
    )

    input_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    total_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    noncritical_variations = 0
    critical_conflicts = 0
    conflict_samples: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    try:
        with backup.open("rb") as reader, temporary.open("wb") as writer:
            for line_number, raw_line in enumerate(reader, start=1):
                input_hash.update(raw_line)
                if not raw_line.strip():
                    continue
                total_rows += 1
                try:
                    row = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    invalid_rows.append({"line": line_number, "error": str(error)})
                    break
                if not isinstance(row, dict):
                    invalid_rows.append({"line": line_number, "error": "JSON object değil"})
                    break
                key = tuple(str(row.get(field, "")) for field in KEY_FIELDS)
                if any(not value for value in key):
                    invalid_rows.append(
                        {"line": line_number, "error": "checkpoint anahtarı eksik", "key": key}
                    )
                    break
                critical_hash = stable_digest(selected_payload(row, CRITICAL_FIELDS))
                diagnostic_hash = stable_digest(non_operational_payload(row))
                existing = connection.execute(
                    """
                    SELECT critical_hash, diagnostic_hash FROM seen
                    WHERE direction=? AND query_variant=? AND search_mode=?
                      AND model_id=? AND query_id=?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    duplicate_rows += 1
                    if existing[0] != critical_hash:
                        critical_conflicts += 1
                        if len(conflict_samples) < 20:
                            conflict_samples.append(
                                {"line": line_number, "key": list(key), "kind": "critical"}
                            )
                    elif existing[1] != diagnostic_hash:
                        noncritical_variations += 1
                    continue

                connection.execute(
                    "INSERT INTO seen VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (*key, critical_hash, diagnostic_hash),
                )
                normalized = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                writer.write(normalized)
                output_hash.update(normalized)
                unique_rows += 1
                if unique_rows % 50_000 == 0:
                    connection.commit()
                    print(
                        f"DEDUP PROGRESS | okunan={total_rows:,} | benzersiz={unique_rows:,} "
                        f"| yinelenen={duplicate_rows:,}",
                        flush=True,
                    )
            connection.commit()

        report = {
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "run_dir": str(run_dir),
            "source": str(source),
            "backup": str(backup),
            "input_bytes": backup.stat().st_size,
            "input_sha256": input_hash.hexdigest(),
            "total_rows": total_rows,
            "unique_rows": unique_rows,
            "duplicate_rows_removed": duplicate_rows,
            "noncritical_duplicate_variations": noncritical_variations,
            "critical_conflicts": critical_conflicts,
            "invalid_rows": invalid_rows,
            "conflict_samples": conflict_samples,
            "output_bytes": temporary.stat().st_size,
            "output_sha256": output_hash.hexdigest(),
            "applied": False,
        }
        if invalid_rows or critical_conflicts:
            report["status"] = "aborted"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise RuntimeError(
                "Onarım uygulanmadı: geçersiz satır veya kritik içerik çatışması bulundu. "
                f"Rapor: {report_path}"
            )

        os.replace(temporary, source)
        report["applied"] = True
        report["status"] = "deduplicated"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {**report, "report": str(report_path)}
    finally:
        connection.close()
        database.unlink(missing_ok=True)
        database.with_name(database.name + "-wal").unlink(missing_ok=True)
        database.with_name(database.name + "-shm").unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Benchmark results.jsonl dosyasını kopya-first ve anahtar-temelli temizler."
    )
    result.add_argument("run_dir", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = repair_results(args.run_dir)
    print("DEDUP_REPORT_JSON:", json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
