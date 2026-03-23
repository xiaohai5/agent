"""Knowledge base service backed by Chroma."""

from __future__ import annotations

import hashlib
import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from . import config_data as config
except ImportError:
    import config_data as config


_DEFAULT_CHROMA_BATCH_SIZE = 5000


def _build_md5_key(md5_str: str, scope: str | None = None) -> str:
    if scope:
        return f"{scope}:{md5_str}"
    return md5_str


def check_md5(md5_str: str, scope: str | None = None) -> bool:
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False

    target = _build_md5_key(md5_str, scope)
    with open(config.md5_path, "r", encoding="utf-8") as f:
        return any(line.strip() == target for line in f.readlines())


def save_md5(md5_str: str, scope: str | None = None) -> None:
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(_build_md5_key(md5_str, scope) + "\n")


def get_md5(docs: list[Document], encoding: str = "utf-8") -> str:
    md5_obj = hashlib.md5()
    for doc in docs:
        md5_obj.update((doc.page_content or "").encode(encoding, errors="ignore"))
    return md5_obj.hexdigest()


class KnowledgeBaseServce(object):
    """Chroma knowledge base service."""

    def __init__(self, splitter_params: dict | None = None, collection_name: str | None = None) -> None:
        params = config.get_splitter_params(splitter_params)
        self.params = params
        self.collection_name = collection_name or config.collection_name
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=OpenAIEmbeddings(model=config.embedding_model),
            persist_directory="./chroma_db",
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.chunk_size,
            chunk_overlap=params.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def _get_batch_size(self) -> int:
        client = getattr(self.chroma, "_client", None)
        max_batch_size = getattr(client, "max_batch_size", None)
        if isinstance(max_batch_size, int) and max_batch_size > 0:
            return max_batch_size
        return _DEFAULT_CHROMA_BATCH_SIZE

    def _add_documents_in_batches(self, docs: list[Document]) -> None:
        batch_size = self._get_batch_size()
        for start in range(0, len(docs), batch_size):
            self.chroma.add_documents(docs[start : start + batch_size])

    def upload_by_str(self, docs: list[Document]) -> str:
        md5_res = get_md5(docs)
        # Scope duplicate detection to the current collection so users do not block each other.
        if check_md5(md5_res, scope=self.collection_name):
            return "Document already exists."

        if len(docs) > self.params.max_splite_num:
            chunks = self.splitter.split_documents(docs)
        else:
            chunks = docs

        self._add_documents_in_batches(chunks)
        save_md5(md5_res, scope=self.collection_name)
        return "Document embedded successfully."

    def list_uploaded_filenames(self) -> list[str]:
        """Return unique filenames stored in the current collection."""
        payload = self.chroma.get(include=["metadatas"])
        metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
        filenames: set[str] = set()
        for meta in metadatas:
            if not isinstance(meta, dict):
                continue
            name = meta.get("filename") or meta.get("source")
            if name:
                filenames.add(str(name))
        return sorted(filenames)
