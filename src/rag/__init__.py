"""RAG mínimo: fragmenta documentos Markdown y recupera los más relevantes con TF-IDF."""
from .retriever import Chunk, Retriever, load_chunks

__all__ = ["Chunk", "Retriever", "load_chunks"]
