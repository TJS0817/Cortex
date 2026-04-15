from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class IngestionResult:
    root_documents: int
    chunks_indexed: int


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _should_use_markdown_splitter(document: Document) -> bool:
    source = str(document.metadata.get("source", "")).lower()
    doc_type = str(document.metadata.get("type", "")).lower()
    return doc_type == "github_readme" or source.endswith(".md") or "readme" in source


def _split_documents(documents: Sequence[Document]) -> List[Document]:
    markdown_docs: List[Document] = []
    standard_docs: List[Document] = []

    for doc in documents:
        if _should_use_markdown_splitter(doc):
            markdown_docs.append(doc)
        else:
            standard_docs.append(doc)

    splits: List[Document] = []

    if standard_docs:
        standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits.extend(standard_splitter.split_documents(standard_docs))

    if markdown_docs:
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=1100,
            chunk_overlap=150,
        )
        splits.extend(markdown_splitter.split_documents(markdown_docs))

    for idx, chunk in enumerate(splits):
        chunk.metadata = dict(chunk.metadata)
        chunk.metadata.setdefault("chunk_index", idx)

    return splits


def _chunk_id(document: Document, index: int) -> str:
    source = str(document.metadata.get("source", "unknown"))
    chunk_text = document.page_content.strip()
    digest = hashlib.sha256(f"{source}|{index}|{chunk_text}".encode("utf-8")).hexdigest()
    return digest


def create_vectorstore(documents: List[Document], reset: bool = False) -> IngestionResult:
    if reset:
        clear_vectorstore()

    splits = _split_documents(documents)
    embeddings = get_embeddings()

    if not splits:
        return IngestionResult(root_documents=len(documents), chunks_indexed=0)

    ids = [_chunk_id(chunk, idx) for idx, chunk in enumerate(splits)]

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )

    vectorstore.add_documents(documents=splits, ids=ids)

    return IngestionResult(root_documents=len(documents), chunks_indexed=len(splits))


def load_vectorstore() -> Optional[Chroma]:
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    try:
        embeddings = get_embeddings()
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
    except Exception:
        return None


def get_retriever() -> Optional[BaseRetriever]:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_db_stats() -> Dict[str, Any]:
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return {"document_chunks": 0, "status": "empty"}

    try:
        records = vectorstore.get(include=[])
        count = len(records.get("ids", []))
        return {"document_chunks": count, "status": "ready"}
    except Exception as exc:
        return {"document_chunks": 0, "status": f"error: {exc}"}


def clear_vectorstore() -> Tuple[bool, str]:
    try:
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        return True, "Vector database was reset successfully."
    except Exception as exc:
        return False, f"Failed to clear vector database: {exc}"
