"""Local catalogue of datasets and extractions (DuckDB, one file, no server)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id VARCHAR PRIMARY KEY,
        name VARCHAR, version VARCHAR, licence_id VARCHAR, source_url VARCHAR,
        registered_at VARCHAR, info_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dataset_files (
        dataset_id VARCHAR, file_name VARCHAR, path VARCHAR, sha256 VARCHAR, size_bytes BIGINT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS extractions (
        extraction_id VARCHAR PRIMARY KEY,
        name VARCHAR, dataset_id VARCHAR, request_hash VARCHAR, created_at VARCHAR,
        jardik_version VARCHAR, output_dir VARCHAR, manifest_path VARCHAR,
        n_buildings BIGINT, n_rows BIGINT, request_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        kind VARCHAR, name VARCHAR, created_at VARCHAR, extraction_id VARCHAR,
        config_hash VARCHAR, output_dir VARCHAR, gate_passed BOOLEAN,
        challenger VARCHAR, best_baseline VARCHAR, metrics_json VARCHAR, config_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS models (
        run_id VARCHAR, model_name VARCHAR, kind VARCHAR, path VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS extraction_buildings (
        extraction_id VARCHAR, building_id VARCHAR, site_id VARCHAR,
        usage_category VARCHAR, rank BIGINT
    )
    """,
)


class Catalogue:
    """Use as a context manager. Re-registering the same id replaces the previous entry."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._con: duckdb.DuckDBPyConnection | None = duckdb.connect(str(path))
        for statement in _SCHEMA:
            self._con.execute(statement)

    # --- lifecycle -----------------------------------------------------------------------
    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> Catalogue:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("The catalogue is closed")
        return self._con

    def _rows(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self.con.execute(query, params or [])
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    # --- datasets ------------------------------------------------------------------------
    def register_dataset(self, info: dict[str, Any], files: list[dict[str, Any]]) -> None:
        dataset_id = info["dataset_id"]
        self.con.execute("DELETE FROM dataset_files WHERE dataset_id = ?", [dataset_id])
        self.con.execute("DELETE FROM datasets WHERE dataset_id = ?", [dataset_id])
        self.con.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                dataset_id,
                info.get("name"),
                info.get("version"),
                info.get("licence_id"),
                info.get("source_url"),
                datetime.now(UTC).isoformat(),
                json.dumps(info, default=str),
            ],
        )
        for f in files:
            self.con.execute(
                "INSERT INTO dataset_files VALUES (?, ?, ?, ?, ?)",
                [dataset_id, f["name"], f["path"], f["sha256"], f["size_bytes"]],
            )

    def list_datasets(self) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT dataset_id, name, version, licence_id, source_url, registered_at "
            "FROM datasets ORDER BY dataset_id"
        )

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM datasets WHERE dataset_id = ?", [dataset_id])
        if not rows:
            return None
        row = rows[0]
        row["info"] = json.loads(row.pop("info_json"))
        return row

    def dataset_files(self, dataset_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT file_name AS name, path, sha256, size_bytes FROM dataset_files "
            "WHERE dataset_id = ? ORDER BY file_name",
            [dataset_id],
        )

    # --- extractions ---------------------------------------------------------------------
    def register_extraction(self, manifest: dict[str, Any]) -> None:
        extraction_id = manifest["extraction_id"]
        output_dir = manifest["output_dir"]
        self.con.execute(
            "DELETE FROM extraction_buildings WHERE extraction_id = ?", [extraction_id]
        )
        self.con.execute("DELETE FROM extractions WHERE extraction_id = ?", [extraction_id])
        counts = manifest.get("counts", {})
        self.con.execute(
            "INSERT INTO extractions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                extraction_id,
                manifest["name"],
                manifest["dataset_id"],
                manifest["request_hash"],
                manifest["created_at"],
                manifest.get("jardik_version"),
                output_dir,
                str(Path(output_dir) / "extraction_manifest.json"),
                counts.get("buildings"),
                counts.get("load_rows"),
                json.dumps(manifest.get("request", {}), default=str),
            ],
        )
        for b in manifest.get("selected_buildings", []):
            self.con.execute(
                "INSERT INTO extraction_buildings VALUES (?, ?, ?, ?, ?)",
                [
                    extraction_id,
                    b["building_id"],
                    b.get("site_id"),
                    b.get("usage_category"),
                    b["rank"],
                ],
            )

    def list_extractions(self) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT extraction_id, name, dataset_id, created_at, n_buildings, n_rows, output_dir "
            "FROM extractions ORDER BY created_at DESC, extraction_id"
        )

    def get_extraction(self, extraction_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM extractions WHERE extraction_id = ?", [extraction_id])
        if not rows:
            return None
        row = rows[0]
        row["request"] = json.loads(row.pop("request_json"))
        return row

    def extraction_buildings(self, extraction_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT building_id, site_id, usage_category, rank FROM extraction_buildings "
            "WHERE extraction_id = ? ORDER BY rank",
            [extraction_id],
        )

    # --- forecast runs -------------------------------------------------------------------
    def register_run(self, manifest: dict[str, Any]) -> None:
        run_id = manifest["run_id"]
        gate = manifest.get("gate", {})
        self.con.execute("DELETE FROM models WHERE run_id = ?", [run_id])
        self.con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                manifest["kind"],
                manifest["name"],
                manifest["created_at"],
                manifest["extraction_id"],
                manifest["config_hash"],
                manifest["output_dir"],
                gate.get("passed"),
                gate.get("challenger"),
                gate.get("best_baseline"),
                json.dumps(manifest.get("metrics", []), default=str),
                json.dumps(manifest.get("config", {}), default=str),
            ],
        )
        for model in manifest.get("models", []):
            self.con.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?)",
                [run_id, model["name"], model.get("kind"), model.get("path")],
            )

    def list_runs(self, extraction_id: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT run_id, kind, name, created_at, extraction_id, gate_passed, challenger, "
            "best_baseline, output_dir FROM runs"
        )
        params: list[Any] = []
        if extraction_id is not None:
            query += " WHERE extraction_id = ?"
            params.append(extraction_id)
        return self._rows(query + " ORDER BY created_at DESC, run_id", params)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM runs WHERE run_id = ?", [run_id])
        if not rows:
            return None
        row = rows[0]
        row["metrics"] = json.loads(row.pop("metrics_json"))
        row["config"] = json.loads(row.pop("config_json"))
        return row

    def run_models(self, run_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT model_name AS name, kind, path FROM models "
            "WHERE run_id = ? ORDER BY model_name",
            [run_id],
        )
