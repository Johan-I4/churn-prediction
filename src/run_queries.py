"""Ejecuta las consultas de sql/queries/ y muestra sus resultados.

Uso:
    python -m src.run_queries              # todas
    python -m src.run_queries --query 02 05  # solo algunas (por número)
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from .build_db import DEFAULT_DB_PATH

QUERIES_DIR = Path(__file__).resolve().parents[1] / "sql" / "queries"


def query_files(queries_dir: str | Path = QUERIES_DIR) -> list[Path]:
    return sorted(Path(queries_dir).glob("[0-9][0-9]_*.sql"))


def header_lines(path: Path) -> list[str]:
    """Líneas de comentario (-- ...) al inicio del archivo: título y explicación."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("--"):
            lines.append(line)
        elif line.strip():
            break
    return lines


def run_query(conn: sqlite3.Connection, path: Path) -> pd.DataFrame:
    return pd.read_sql_query(path.read_text(encoding="utf-8"), conn)


def main(argv=None):
    p = argparse.ArgumentParser(description="Ejecuta las consultas SQL del proyecto")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--query", nargs="+", help="números de consulta, por ejemplo: 01 05")
    args = p.parse_args(argv)

    if not Path(args.db).exists():
        raise SystemExit(f"No existe {args.db}. Créala primero con: python -m src.build_db")

    files = query_files()
    if args.query:
        wanted = {q.zfill(2) for q in args.query}
        files = [f for f in files if f.name[:2] in wanted]
        if not files:
            raise SystemExit(f"Ninguna consulta coincide con {args.query}")

    conn = sqlite3.connect(args.db)
    try:
        for path in files:
            print("\n" + "=" * 78)
            print("\n".join(header_lines(path)))
            print("-" * 78)
            print(run_query(conn, path).to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
