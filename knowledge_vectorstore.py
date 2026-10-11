"""向量存储模块 — 基于 Chroma + 多后端 Embedding (Ollama / DeepSeek / DashScope)"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

EMBED_BACKEND = os.environ.get("HONALA_EMBED_BACKEND", "ollama").lower()

OLLAMA_EMBED_URL = os.environ.get("HONALA_OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings")
OLLAMA_EMBED_MODEL = os.environ.get("HONALA_OLLAMA_EMBED_MODEL", "nomic-embed-text")

DEEPSEEK_EMBED_URL = "https://api.deepseek.com/v1/embeddings"
DEEPSEEK_EMBED_MODEL = "deepseek-chat"

DASHSCOPE_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
DASHSCOPE_EMBED_MODEL = os.environ.get("HONALA_DASHSCOPE_EMBED_MODEL", "text-embedding-v4")

EMBED_TIMEOUT = int(os.environ.get("HONALA_EMBED_TIMEOUT", "60"))

DATA_DIR = Path(os.environ.get("HONALA_KNOWLEDGE_DATA_DIR", os.path.join(os.path.dirname(__file__), "knowledge_data"))).resolve()
CHROMA_DIR = DATA_DIR / "chroma_db"
DOC_META_FILE = DATA_DIR / "documents.json"
UPLOAD_DIR = DATA_DIR / "uploads"


def _load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "free_llm_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _get_api_key(provider: str) -> str:
    cfg = _load_config()
    if provider == "dashscope":
        key = cfg.get("dashscope_api_key", "")
        if key: return key
        return os.environ.get("HONALA_DASHSCOPE_API_KEY", "")
    if provider == "deepseek":
        key = cfg.get("api_key", "")
        if key: return key
        return os.environ.get("HONALA_DEEPSEEK_API_KEY", "")
    return ""


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        return [0.0] * 768

    if EMBED_BACKEND == "deepseek":
        return _embed_deepseek(text)
    elif EMBED_BACKEND == "dashscope":
        return _embed_dashscope(text)
    else:
        return _embed_ollama(text)


def _embed_ollama(text: str) -> list[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("embedding", [])
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama embedding 不可达: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Ollama embedding 失败: {e}")


def _embed_openai_compatible(url: str, api_key: str, model: str, text: str) -> list[float]:
    payload = {"model": model, "input": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [{}])[0].get("embedding", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"API embedding HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"API embedding 不可达: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"API embedding 失败: {e}")


def _embed_deepseek(text: str) -> list[float]:
    api_key = _get_api_key("deepseek")
    if not api_key:
        raise RuntimeError("DeepSeek API key 未配置 (free_llm_config.json 中设置 api_key)")
    return _embed_openai_compatible(DEEPSEEK_EMBED_URL, api_key, DEEPSEEK_EMBED_MODEL, text)


def _embed_dashscope(text: str) -> list[float]:
    api_key = _get_api_key("dashscope")
    if not api_key:
        raise RuntimeError("DashScope API key 未配置 (free_llm_config.json 中设置 dashscope_api_key)")
    return _embed_openai_compatible(DASHSCOPE_EMBED_URL, api_key, DASHSCOPE_EMBED_MODEL, text)


def get_collection():
    """获取 Chroma collection"""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="rag_knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )


def _load_doc_meta() -> dict[str, Any]:
    if DOC_META_FILE.exists():
        try:
            return json.loads(DOC_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_doc_meta(meta: dict[str, Any]):
    DOC_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def index_document(
    doc_id: str,
    filename: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """将文档的 chunks 向量化并存入 Chroma"""
    ensure_dirs()
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    embeddings = []
    for text in texts:
        try:
            emb = get_embedding(text)
            embeddings.append(emb)
        except Exception:
            embeddings.append([0.0] * 768)
        time.sleep(0.05)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    meta = _load_doc_meta()
    meta[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _save_doc_meta(meta)

    return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}


def search_similar(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """向量相似度检索"""
    ensure_dirs()
    collection = get_collection()

    try:
        query_embedding = get_embedding(query)
    except Exception:
        return []

    if not query_embedding:
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    hits = []
    for i in range(len(docs)):
        dist = dists[i] if i < len(dists) else 1.0
        score = float(1.0 - dist) if dist <= 1.0 else float(1.0 / (1.0 + dist))
        if score < score_threshold:
            continue
        hits.append({
            "text": docs[i],
            "score": round(score, 4),
            "doc_id": metas[i].get("doc_id", "") if i < len(metas) else "",
            "filename": metas[i].get("filename", "") if i < len(metas) else "",
            "chunk_index": metas[i].get("chunk_index", 0) if i < len(metas) else 0,
        })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


def list_documents() -> list[dict[str, Any]]:
    meta = _load_doc_meta()
    return sorted(meta.values(), key=lambda d: d.get("indexed_at", ""), reverse=True)


def delete_document(doc_id: str) -> dict[str, Any]:
    collection = get_collection()
    try:
        results = collection.get(where={"doc_id": doc_id})
        if results and results.get("ids"):
            collection.delete(ids=results["ids"])
    except Exception:
        pass

    meta = _load_doc_meta()
    removed = meta.pop(doc_id, None)
    _save_doc_meta(meta)

    return {"doc_id": doc_id, "deleted": removed is not None, "filename": removed.get("filename", "") if removed else ""}


def _current_embed_model() -> str:
    if EMBED_BACKEND == "deepseek":
        return DEEPSEEK_EMBED_MODEL
    elif EMBED_BACKEND == "dashscope":
        return DASHSCOPE_EMBED_MODEL
    return OLLAMA_EMBED_MODEL


def get_collection_stats() -> dict[str, Any]:
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0
    meta = _load_doc_meta()
    return {
        "chunk_count": count,
        "document_count": len(meta),
        "embed_backend": EMBED_BACKEND,
        "embed_model": _current_embed_model(),
        "documents": list(meta.values()),
    }
