"""知识库 API 端点 — FastAPI Router"""
import json
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from utils import utc_iso
from knowledge_loader import load_document, get_supported_extensions
from knowledge_chunker import chunk_text
from knowledge_vectorstore import (
    index_document,
    search_similar,
    list_documents,
    delete_document,
    get_collection_stats,
    get_embedding,
    UPLOAD_DIR,
    EMBED_BACKEND,
)

router = APIRouter(prefix="/knowledge")

OLLAMA_CHAT_URL = os.environ.get("HONALA_OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
CHAT_TIMEOUT = max(5, int(os.environ.get("HONALA_CHAT_TIMEOUT_SECONDS", "60")))
DEEPSEEK_API_URL = os.environ.get("HONALA_DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions").strip()


def _chunk_text_resp(msg: str) -> list[str]:
    clean = msg.replace("\r\n", "\n")
    return [clean[i:i + 48] for i in range(0, len(clean), 48)] or [""]


def _make_stream_line(content: str = "", done: bool = False, model: str = "rag", meta: dict[str, Any] | None = None) -> bytes:
    payload: dict[str, Any] = {"model": model, "created_at": utc_iso(), "message": {"role": "assistant", "content": content}, "done": done}
    if meta:
        payload.update(meta)
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _synthetic_stream(message: str) -> StreamingResponse:
    def generate():
        for part in _chunk_text_resp(message):
            if part:
                yield _make_stream_line(part)
        yield _make_stream_line("", done=True)
    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── 文档上传 ────────────────────────────

@router.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...), chunk_size: int = 512, chunk_overlap: int = 64):
    """上传一个或多个文档，自动解析、分块、索引"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    errors = []

    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in get_supported_extensions():
            errors.append({"filename": file.filename, "error": f"不支持的文件格式: {ext}"})
            continue

        doc_id = uuid.uuid4().hex[:12]
        save_path = UPLOAD_DIR / f"{doc_id}{ext}"
        try:
            content = await file.read()
            save_path.write_bytes(content)
        except Exception as e:
            errors.append({"filename": file.filename, "error": f"写入文件失败: {e}"})
            continue

        if save_path.stat().st_size > 50 * 1024 * 1024:
            save_path.unlink(missing_ok=True)
            errors.append({"filename": file.filename, "error": "文件超过 50MB 限制"})
            continue

        text = load_document(str(save_path))
        if not text or not text.strip():
            errors.append({"filename": file.filename, "error": "文档内容为空或解析失败"})
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            errors.append({"filename": file.filename, "error": "文档分块后无有效内容"})
            continue

        try:
            result = index_document(doc_id, file.filename, chunks)
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "error": f"索引失败: {e}"})

    return JSONResponse({
        "ok": len(results) > 0,
        "indexed": results,
        "errors": errors,
        "total": len(files),
        "success_count": len(results),
        "error_count": len(errors),
    })


# ── 知识库搜索 ────────────────────────────

@router.post("/search")
async def search_knowledge(request: Request):
    """向量语义检索"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    query = str(body.get("query") or "").strip()[:1000]
    if not query:
        return JSONResponse({"error": "missing_query"}, status_code=400)

    top_k = min(max(int(body.get("top_k", 5)), 1), 20)
    score_threshold = float(body.get("score_threshold", 0.0))

    try:
        hits = search_similar(query, top_k=top_k, score_threshold=score_threshold)
    except Exception as e:
        return JSONResponse({"error": f"检索失败: {e}", "results": []}, status_code=500)

    if not hits:
        return JSONResponse({"results": [], "query": query, "hint": "未找到相关文档，请尝试调整查询关键词"})

    formatted_results = []
    for h in hits:
        formatted_results.append({
            "text": h["text"],
            "score": h["score"],
            "doc_id": h["doc_id"],
            "filename": h["filename"],
            "chunk_index": h["chunk_index"],
        })

    return JSONResponse({"results": formatted_results, "query": query, "count": len(formatted_results)})


# ── RAG 问答 (知识库模式) ────────────────────

@router.post("/chat")
async def rag_chat(request: Request):
    """知识库问答：检索相关文档 → 组装 Prompt → LLM 生成答案"""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)

    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = str(msg.get("content", "")).strip()[:1000]
            break
    if not user_query:
        return _synthetic_stream("未能从消息中提取用户问题。")

    top_k = payload.get("top_k", 5)
    try:
        hits = search_similar(user_query, top_k=top_k)
    except Exception as e:
        return _synthetic_stream(f"知识库检索失败: {e}")

    if not hits:
        return _synthetic_stream("知识库中未找到与您问题相关的文档内容，请尝试更具体的提问。")

    context_parts = []
    sources = []
    for i, h in enumerate(hits, 1):
        context_parts.append(f"[文档{i}] (来自: {h['filename']}, 相关度: {h['score']})\n{h['text']}")
        sources.append({"index": i, "filename": h["filename"], "score": h["score"], "text": h["text"][:200]})

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "你是一个基于知识库的智能问答助手。请根据以下知识库文档内容回答用户问题。\n\n"
        "回答规则：\n"
        "1. 优先使用知识库中的信息回答问题，引用时使用 [文档N] 标注来源\n"
        "2. 如果知识库内容不足以回答问题，请明确说明\n"
        "3. 回答结尾列出参考来源\n\n"
        f"知识库内容：\n{context}\n\n"
    )

    grounded_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.get("role") != "system":
            grounded_messages.append(msg)

    stream = bool(payload.get("stream", True))
    if not stream:
        try:
            result = _call_deepseek_nonstream(grounded_messages, payload)
            result["sources"] = sources
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e), "sources": sources}, status_code=500)

    return _relay_ollama_stream(grounded_messages, payload, sources)


def _call_deepseek_nonstream(messages: list[dict], payload: dict) -> dict:
    api_key = _load_config_value("api_key", "HONALA_DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DeepSeek API key not configured")
    model = payload.get("model", "deepseek-chat")
    req_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": float(payload.get("temperature", 0.7)),
        "top_p": float(payload.get("top_p", 0.9)),
    }
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=CHAT_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {"content": data.get("choices", [{}])[0].get("message", {}).get("content", ""), "model": model}


def _relay_ollama_stream(messages: list[dict], original_payload: dict, sources: list) -> StreamingResponse:
    payload = {
        "model": original_payload.get("model", "qwen3"),
        "messages": messages,
        "stream": True,
    }
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=CHAT_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse({"error": "ollama_error", "detail": body[:500]}, status_code=exc.code)
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "ollama_unreachable", "detail": str(exc.reason)}, status_code=502)

    def generate():
        try:
            while True:
                chunk = upstream.readline()
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()
            yield _make_stream_line("", done=True, meta={"sources": sources})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── 文档管理 ────────────────────────────

@router.get("/documents")
async def api_list_documents():
    stats = get_collection_stats()
    return JSONResponse(stats)


@router.delete("/documents/{doc_id}")
async def api_delete_document(doc_id: str):
    result = delete_document(doc_id)
    return JSONResponse(result)


# ── 知识库状态 ────────────────────────────

@router.get("/status")
async def api_status():
    """检查 Embedding 后端和 Chroma 连通状态"""
    embed_ok = False
    try:
        test_embed = get_embedding("test")
        embed_ok = len(test_embed) > 0
    except Exception:
        pass

    stats = get_collection_stats()
    return JSONResponse({
        "embed_ok": embed_ok,
        "embed_backend": stats.get("embed_backend", EMBED_BACKEND),
        "embed_model": stats.get("embed_model", ""),
        "document_count": stats["document_count"],
        "chunk_count": stats["chunk_count"],
    })


# ── 配置读取辅助 ────────────────────────

def _load_config_value(key: str, env_name: str = "", default: str = "") -> str:
    config_path = os.path.join(os.path.dirname(__file__), "free_llm_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                val = cfg.get(key, "")
                if val and str(val).strip():
                    return str(val).strip()
    except Exception:
        pass
    if env_name:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    return default
