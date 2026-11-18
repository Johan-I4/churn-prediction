import sqlite3

import pytest

from src.build_db import build_db
from src.data import load_clean
from src.run_queries import header_lines, query_files, run_query


@pytest.fixture(scope="module")
def db_path(synthetic_csv, tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "churn.db"
    build_db(synthetic_csv, path)
    return path


@pytest.fixture()
def conn(db_path):
    c = sqlite3.connect(db_path)
    yield c
    c.close()


def query_by_number(number: str):
    return next(p for p in query_files() if p.name.startswith(number))


def test_three_tables_have_same_row_count(conn):
    counts = [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ("clientes", "servicios", "facturacion")]
    assert counts == [800, 800, 800]


def test_foreign_keys_are_consistent(conn):
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_total_charges_blanks_were_fixed(conn):
    assert conn.execute("SELECT COUNT(*) FROM facturacion WHERE total_charges IS NULL").fetchone()[0] == 0


def test_at_least_10_queries():
    assert len(query_files()) >= 10


@pytest.mark.parametrize("path", query_files(), ids=lambda p: p.name)
def test_each_query_has_title_and_3_explanation_lines(path):
    lines = header_lines(path)
    assert lines[0].startswith("-- Consulta"), "la primera línea debe ser el título"
    for prefix in ("-- Pregunta:", "-- Qué hace:", "-- Cómo leerla:"):
        assert any(line.startswith(prefix) for line in lines), f"{path.name}: falta '{prefix}'"


@pytest.mark.parametrize("path", query_files(), ids=lambda p: p.name)
def test_each_query_runs(conn, path):
    df = run_query(conn, path)
    assert len(df.columns) >= 1  # con datos sintéticos algunas consultas pueden dar 0 filas


def test_queries_cover_required_techniques():
    text = "\n".join(p.read_text(encoding="utf-8").upper() for p in query_files())
    for keyword in ("JOIN", "LEFT JOIN", "HAVING", "WITH ", " OVER ("):
        assert keyword in text, f"ninguna consulta usa {keyword.strip()}"


def test_global_churn_matches_pandas(conn, synthetic_csv):
    df = run_query(conn, query_by_number("01"))
    expected = load_clean(synthetic_csv)["Churn"]
    assert int(df.loc[0, "clientes"]) == len(expected)
    assert int(df.loc[0, "bajas"]) == int(expected.sum())
    # SQLite y Python redondean distinto los empates (38.625), por eso la tolerancia
    assert df.loc[0, "pct_churn"] == pytest.approx(expected.mean() * 100, abs=0.01)


def test_churn_by_contract_matches_pandas(conn, synthetic_csv):
    sql = run_query(conn, query_by_number("02")).set_index("contract")["pct_churn"]
    pandas_ = load_clean(synthetic_csv).groupby("Contract")["Churn"].mean().mul(100).round(2)
    for contract, value in pandas_.items():
        assert sql[contract] == pytest.approx(value, abs=0.01)


def test_integrity_query_finds_no_orphans(conn):
    row = run_query(conn, query_by_number("07")).iloc[0]
    assert (row["sin_facturacion"], row["sin_servicios"]) == (0, 0)


def test_missing_csv_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_db(tmp_path / "no_existe.csv", tmp_path / "x.db")
