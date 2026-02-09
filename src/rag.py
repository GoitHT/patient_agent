"""RAG (检索增强生成) 模块 - 合并了知识库加载、索引构建和检索功能"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


# ============================================================================
# 知识库加载 (原 rag/loaders.py)
# ============================================================================

@dataclass(frozen=True)
class KBChunk:
    id: str
    text: str
    meta: dict[str, Any]


def _chunk_text(text: str, *, max_chars: int = 600, overlap: int = 80) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunks.append("\n\n".join(buf).strip())
        buf = []
        buf_len = 0

    for p in paragraphs:
        if buf_len + len(p) + 2 <= max_chars:
            buf.append(p)
            buf_len += len(p) + 2
            continue
        flush()
        if len(p) <= max_chars:
            buf.append(p)
            buf_len = len(p)
            continue

        start = 0
        while start < len(p):
            end = min(len(p), start + max_chars)
            chunks.append(p[start:end].strip())
            start = end - overlap if end - overlap > start else end
        flush()

    flush()
    return chunks


def _infer_dept(rel_parts: tuple[str, ...]) -> str:
    if not rel_parts:
        return "unknown"
    return rel_parts[0]


def _infer_type(stem: str) -> str:
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "misc"


def load_kb_chunks(kb_root: Path) -> list[KBChunk]:
    kb_root = kb_root.resolve()
    chunks: list[KBChunk] = []

    for path in sorted(kb_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        rel = path.relative_to(kb_root)
        rel_parts = tuple(rel.parts)
        dept = _infer_dept(rel_parts)
        typ = _infer_type(path.stem)
        doc_id = str(rel).replace("\\", "/")
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        source = str(path).replace("\\", "/")

        text = path.read_text(encoding="utf-8")
        for idx, chunk_text in enumerate(_chunk_text(text)):
            chunk_id = f"{idx:04d}"
            chunk_uid = f"{doc_id}::chunk:{chunk_id}"
            meta = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "dept": dept,
                "type": typ,
                "source": source,
                "updated_at": updated_at,
            }
            chunks.append(KBChunk(id=chunk_uid, text=chunk_text, meta=meta))

    return chunks


# ============================================================================
# 嵌入函数和检索器 (原 rag/retriever.py)
# ============================================================================

class HashEmbeddingFunction:
    """Deterministic local embedding (no network, no ML model).

    This is not semantically perfect, but sufficient for deterministic
    unit tests + traceable retrieval in this simulation project.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hash-embedding-v1"

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(dim=int(config.get("dim", 384)))

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: list[str]) -> list[list[float]]:  # chromadb protocol
        return [self._embed_one(t).tolist() for t in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def _tokenize(self, text: str) -> list[str]:
        text = (text or "").lower()
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
        return tokens

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in self._tokenize(text):
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="little") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "meta": self.meta,
        }


class ChromaRetriever:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str = "hospital_kb",
        dim: int = 384,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding = HashEmbeddingFunction(dim=dim)
        
        # 禁用遥测
        settings = chromadb.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir), settings=settings)
        self.collection = self.client.get_collection(
            name=self.collection_name, embedding_function=self.embedding
        )

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        where = None
        if filters:
            clauses: list[dict[str, Any]] = []
            for key in ("dept", "type"):
                if key in filters and filters[key] is not None:
                    clauses.append({key: filters[key]})
            if len(clauses) == 1:
                where = clauses[0]
            elif len(clauses) > 1:
                where = {"$and": clauses}

        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc_text, meta, dist in zip(docs, metas, distances, strict=False):
            score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
            doc_id = (meta or {}).get("doc_id") or "unknown"
            chunk_id = (meta or {}).get("chunk_id") or "0000"
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=score,
                    text=doc_text or "",
                    meta=meta or {},
                )
            )
        return [c.as_dict() for c in chunks]


class DummyRetriever:
    """虚拟检索器 - 用于测试或不需要RAG功能时"""
    
    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        k: int = 4,
    ) -> list[dict[str, Any]]:
        """返回空检索结果"""
        return []


# ============================================================================
# 索引构建 (原 rag/index.py)
# ============================================================================

def build_index(
    *,
    kb_root: Path,
    persist_dir: Path,
    collection_name: str = "hospital_kb",
    dim: int = 384,
) -> dict[str, Any]:
    """Build (or rebuild) a Chroma persistent index from local KB files."""

    kb_root = Path(kb_root)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    settings = chromadb.Settings(anonymized_telemetry=False)
    client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=HashEmbeddingFunction(dim=dim)
    )

    chunks = load_kb_chunks(kb_root)
    if not chunks:
        return {"added": 0, "collection_name": collection_name, "persist_dir": str(persist_dir)}

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.meta for c in chunks],
    )

    return {
        "added": len(chunks),
        "collection_name": collection_name,
        "persist_dir": str(persist_dir),
    }


__all__ = [
    "KBChunk",
    "load_kb_chunks",
    "HashEmbeddingFunction",
    "RetrievedChunk",
    "ChromaRetriever",
    "build_index",
]
