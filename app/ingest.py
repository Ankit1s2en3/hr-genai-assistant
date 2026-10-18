"""Turn an uploaded policy document into embedded chunks in pgvector."""

import io

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ALLOWED_EXTENSIONS = (".pdf", ".txt")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def split_into_documents(filename: str, text: str) -> list[Document]:
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"filename": filename, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]


def ingest_document(filename: str, data: bytes, store: VectorStore) -> int:
    """Store the document's chunks and return how many were stored."""
    docs = split_into_documents(filename, extract_text(filename, data))
    if docs:
        # Stable ids mean re-uploading the same file overwrites its chunks instead of duplicating them.
        ids = [f"{filename}:{doc.metadata['chunk']}" for doc in docs]
        store.add_documents(docs, ids=ids)
    return len(docs)
