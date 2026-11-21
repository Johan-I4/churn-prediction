"""Recuperación de fragmentos con TF-IDF (scikit-learn). No usa embeddings ni base vectorial.

Flujo: leer `docs/knowledge/*.md` -> partir por encabezados `##` -> indexar -> buscar los k
fragmentos más parecidos a la consulta. Solo esos fragmentos llegan al generador de texto,
así no se mete toda la documentación en el contexto.

La carpeta se puede cambiar con la variable de entorno KNOWLEDGE_DIR.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_KNOWLEDGE_DIR = "docs/knowledge"

SPANISH_STOPWORDS = [
    "a", "al", "algo", "con", "como", "cual", "cuando", "de", "del", "el", "ella", "en", "es",
    "esta", "este", "esto", "la", "las", "lo", "los", "me", "mi", "no", "o", "para", "pero",
    "por", "que", "se", "si", "sin", "su", "sus", "te", "un", "una", "uno", "y", "ya",
]


@dataclass(frozen=True)
class Chunk:
    source: str  # nombre del archivo
    title: str  # encabezado `##` del fragmento
    text: str  # texto completo (título incluido)


def knowledge_dir() -> Path:
    return Path(os.getenv("KNOWLEDGE_DIR", DEFAULT_KNOWLEDGE_DIR))


def split_markdown(text: str, source: str) -> list[Chunk]:
    """Un fragmento por cada encabezado `##`. Lo que va antes del primero se descarta."""
    chunks: list[Chunk] = []
    title, lines = None, []

    def flush():
        body = "\n".join(lines).strip()
        if title and body:
            chunks.append(Chunk(source=source, title=title, text=f"{title}\n{body}"))

    for line in text.splitlines():
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            flush()
            title, lines = match.group(1), []
        elif title is not None:
            lines.append(line)
    flush()
    return chunks


def load_chunks(directory: str | Path | None = None) -> list[Chunk]:
    directory = Path(directory) if directory is not None else knowledge_dir()
    if not directory.is_dir():
        raise FileNotFoundError(f"No encontré la carpeta de conocimiento '{directory}'.")
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(split_markdown(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise ValueError(f"'{directory}' no contiene fragmentos (archivos .md con encabezados '##').")
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=SPANISH_STOPWORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> "Retriever":
        return cls(load_chunks(directory))

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        """Los k fragmentos más similares con similitud coseno >= min_score, de mayor a menor."""
        if not query.strip():
            return []
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] >= min_score]
