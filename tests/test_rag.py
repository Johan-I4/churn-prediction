import pytest

from src.rag import Retriever, load_chunks
from src.rag.retriever import split_markdown


def test_split_markdown_one_chunk_per_heading():
    text = "# Título\nintro que se descarta\n\n## A\nuno\n\n## B\ndos\ntres\n"
    chunks = split_markdown(text, "x.md")
    assert [c.title for c in chunks] == ["A", "B"]
    assert chunks[1].text == "B\ndos\ntres"
    assert all(c.source == "x.md" for c in chunks)


def test_real_knowledge_base_loads():
    chunks = load_chunks()
    assert len(chunks) >= 8
    assert {c.source for c in chunks} >= {"umbral_y_modelo.md", "factores_de_riesgo.md", "limitaciones.md"}


@pytest.fixture(scope="module")
def retriever():
    return Retriever.from_directory()


@pytest.mark.parametrize(
    "query, expected_title",
    [
        ("umbral 0.34 maximiza el beneficio neto, cliente perdido vale 500", "Umbral de decisión por costos"),
        ("Contrato Contract Month-to-month Internet InternetService Fiber optic", "Contrato mes a mes (Month-to-month) con fibra óptica (Fiber optic)"),
        ("Método de pago PaymentMethod Electronic check", "Método de pago con cheque electrónico (Electronic check)"),
        ("causalidad asociación", "Asociación, no causalidad"),
    ],
)
def test_top_result_is_the_expected_chunk(retriever, query, expected_title):
    results = retriever.search(query, k=3)
    assert results, "no recuperó nada"
    assert results[0][0].title == expected_title


def test_scores_are_sorted_and_k_is_respected(retriever):
    results = retriever.search("churn contrato antigüedad", k=2)
    assert len(results) <= 2
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_or_empty_query_returns_nothing(retriever):
    assert retriever.search("   ") == []
    assert retriever.search("zzzz qqqq wwww") == []


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "no_existe")
