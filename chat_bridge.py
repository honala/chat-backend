# ADR-001 (2026-05-19): 图片上传"URL 方案"已废弃，回退纯 base64
# 三个失败节点：
#   1. Cloudflare Tunnel 路由 — 境外 edge 节点无法回源到新增 /api/chat-upload，
#      国内正常，路径注册问题
#   2. LLM 取图 — DeepSeek/DashScope 从 Cloudflare 取 ECS 文件失败，网络拓扑不通
#   3. 后端代理转 base64 — resolve_file_urls() 把磁盘文件读出转 base64 再发 LLM，
#      等于绕一圈没绕，方案自我否定
# 结论：单用户场景 base64 够用，不要再次尝试 URL 上传方案。
import ast
import html as _htmlmod
import json
import math
import operator
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from utils import utc_iso
from database import (
    create_session,
    delete_messages,
    delete_session,
    get_messages,
    init_db,
    list_providers,
    list_sessions,
    save_file_meta,
    save_messages,
    save_providers,
    update_session,
)


router = APIRouter()

OLLAMA_CHAT_URL = os.environ.get("HONALA_OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
REQUEST_TIMEOUT = max(5, int(os.environ.get("HONALA_CHAT_TIMEOUT_SECONDS", "60")))
SOURCE_LIMIT = 6
DEEPSEEK_API_URL = os.environ.get("HONALA_DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions").strip()
DASHSCOPE_API_URL = os.environ.get("HONALA_DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions").strip()

VOLCANO_API_URL = os.environ.get("HONALA_VOLCANO_API_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions").strip()

_CFG_CACHE = None

def _load_config() -> dict:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    config_path = os.path.join(os.path.dirname(__file__), "free_llm_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                _CFG_CACHE = json.load(f)
                return _CFG_CACHE
    except Exception:
        pass
    _CFG_CACHE = {}
    return _CFG_CACHE

def _get_config_value(key: str, env_name: str = "", default: str = "") -> str:
    """Read config value — tries config file first, then env var."""
    cfg = _load_config()
    val = cfg.get(key, "")
    if val is not None and val != "" and str(val).strip():
        return str(val).strip()
    if env_name:
        val = os.environ.get(env_name, "").strip()
        if val: return val
    return default

def _get_config_int(key: str, env_name: str = "", default: int = 0) -> int:
    """Read config value as int, with safe fallback."""
    s = _get_config_value(key, env_name)
    try:
        return int(s) if s else default
    except ValueError:
        return default

def _get_api_key(provider: str) -> str:
    if provider == "dashscope":
        key = _get_config_value("dashscope_api_key", "HONALA_DASHSCOPE_API_KEY")
        if key: return key
    if provider == "volc":
        key = _get_config_value("volc_api_key", "HONALA_VOLC_API_KEY")
        if key: return key
    key = _get_config_value("api_key", "HONALA_DEEPSEEK_API_KEY")
    return key

def _get_backend(model: str):
    """Return (api_url, api_key, is_qwen) for a given model.
    Routes through DashScope (百炼) for all supported 百炼 models,
    including Qwen, DeepSeek (via 百炼), Kimi, GLM, MiniMax, etc.
    Falls back to DeepSeek API for unrecognized models.
    """
    model_lower = model.lower()
    # DashScope (百炼) 支持的模型家族
    dashscope_prefixes = (
        "qwen",
        "deepseek-v4",
        "kimi-",
        "glm-",
        "minimax-",
        "mimo-",
    )
    if model_lower.startswith(dashscope_prefixes):
        return DASHSCOPE_API_URL, _get_api_key("dashscope"), True
    # Volcano Engine (方舟) 支持的模型
    if model_lower.startswith("doubao-"):
        return VOLCANO_API_URL, _get_api_key("volc"), True
    return DEEPSEEK_API_URL, _get_api_key("deepseek"), False


def chunk_text(text: str, size: int = 48) -> list[str]:
    clean = text.replace("\r\n", "\n")
    return [clean[i:i + size] for i in range(0, len(clean), size)] or [""]


def make_stream_line(content: str = "", done: bool = False, model: str = "web-bridge", meta: dict[str, Any] | None = None) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "created_at": utc_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if meta:
        payload.update(meta)
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def synthetic_stream(message: str, status_code: int = 200) -> StreamingResponse:
    def generate():
        for part in chunk_text(message):
            if part:
                yield make_stream_line(part)
        yield make_stream_line("", done=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson", status_code=status_code)


def relay_ollama(payload: dict[str, Any]) -> StreamingResponse | JSONResponse:
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson, application/json"},
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse(
            {"error": "ollama_upstream_error", "status": exc.code, "detail": body or exc.reason},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "ollama_unreachable", "detail": str(exc.reason)}, status_code=502)

    stream = bool(payload.get("stream", True))
    media_type = upstream.headers.get_content_type() or "application/x-ndjson"

    if stream:
        def generate():
            try:
                while True:
                    chunk = upstream.readline()
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()

        return StreamingResponse(generate(), media_type=media_type)

    try:
        data = json.loads(upstream.read().decode("utf-8"))
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid_upstream_response"}, status_code=502)
    finally:
        upstream.close()
    return JSONResponse(data)


def extract_search_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if "【附件" in content:
            content = content.split("【附件", 1)[0].rstrip()
        return content[:500]
    return ""


def format_source_lines(sources: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in sources[:SOURCE_LIMIT]:
        title = item.get("title", "").strip() or item.get("url", "").strip() or "Untitled"
        url = item.get("url", "").strip()
        host = item.get("host", "").strip()
        published = item.get("published", "").strip()
        parts = [part for part in (host, published) if part]
        detail = f" ({' · '.join(parts)})" if parts else ""
        if url:
            lines.append(f"- [{title}]({url}){detail}")
        else:
            lines.append(f"- {title}{detail}")
    return "\n".join(lines)


def build_grounding_guidance(query: str, sources: list[dict[str, str]], provider: str) -> str:
    formatted_sources = format_source_lines(sources)
    return (
        f"以下是 {provider} 实时检索到的网页片段，请把它们当作本轮回答的外部依据。\n"
        "回答规则：\n"
        "1. 优先使用较新的、较权威的来源；如果时效性重要，要直接说明日期。\n"
        "2. 不要假装看过未提供的页面；如果上下文不足，要明确说“现有结果不足以确认”。\n"
        "3. 当不同来源冲突时，要显式指出冲突，而不是强行合并。\n"
        "4. 正文里在关键事实句后使用 [1] [2] 这种编号标注来源。\n"
        "5. 回答结尾必须输出一个“## 参考来源”小节，并使用下面这份 Markdown 列表；不要改写 URL。\n"
        f"6. 用户原始检索问题是：{query}\n\n"
        "参考来源模板：\n"
        "## 参考来源\n"
        f"{formatted_sources}\n"
    )


def call_exa_search(query: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
    api_key = _get_config_value("exa_api_key", "HONALA_EXA_API_KEY")
    if not api_key:
        return None, None, []

    num_results = max(1, min(10, _get_config_int("exa_num_results", "", 5)))
    max_chars = max(500, min(12000, _get_config_int("exa_highlight_max_chars", "", 3200)))

    payload = {
        "query": query,
        "type": _get_config_value("exa_search_type", "", "auto"),
        "numResults": num_results,
        "contents": {
            "highlights": {
                "maxCharacters": max_chars,
            }
        },
    }
    req = urllib.request.Request(
        _get_config_value("exa_search_url", "", "https://api.exa.ai/search"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": api_key,
            "User-Agent": "HonalaChatBridge/1.0 (+self-hosted local Ollama chat)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"Exa 联网检索返回 HTTP {exc.code}。{detail[:300]}".strip(), []
    except urllib.error.URLError as exc:
        return None, f"Exa 联网检索不可达：{exc.reason}", []
    except Exception as exc:
        return None, f"Exa 联网检索失败：{exc}", []

    results = data.get("results")
    if not isinstance(results, list):
        return None, "Exa 返回了不可识别的结果结构。", []

    entries: list[str] = []
    source_items: list[dict[str, str]] = []
    for idx, item in enumerate(results[:num_results], start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or f"来源 {idx}").strip()
        published = str(item.get("publishedDate") or "").strip()
        author = str(item.get("author") or "").strip()
        highlights = item.get("highlights")
        if not isinstance(highlights, list):
            highlights = []
        text = str(item.get("text") or "").strip()
        snippet_lines = [str(s).strip() for s in highlights if str(s).strip()]
        if not snippet_lines and text:
            snippet_lines = [text[:600].strip()]
        if not snippet_lines:
            continue

        header_bits = [f"[{idx}] {title}"]
        if published:
            header_bits.append(f"date={published}")
        if author:
            header_bits.append(f"author={author}")

        entry = "\n".join([
            " | ".join(header_bits),
            f"URL: {url}",
            *[f"- {line}" for line in snippet_lines[:6]],
        ])
        entries.append(entry)
        snippet = "\n".join(snippet_lines[:3])
        if url:
            source_items.append(
                {
                    "title": title,
                    "url": url,
                    "host": url.split("/", 3)[2] if "://" in url else "",
                    "published": published[:10] if published else "",
                    "snippet": snippet,
                }
            )

    if not entries:
        return None, "Exa 已执行检索，但没有返回足够可用的网页摘要。", []

    guidance = build_grounding_guidance(query, source_items, "Exa")
    context = "\n\n".join(entries)
    return f"{guidance}\n\nSearch query: {query}\n\n{context}", None, source_items


def call_brave_context(query: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
    api_key = _get_config_value("brave_api_key", "HONALA_BRAVE_API_KEY")
    if not api_key:
        return None, None, []

    payload = {
        "q": query,
        "country": _get_config_value("brave_country", "", "cn"),
        "search_lang": _get_config_value("brave_search_lang", "", "zh-hans"),
        "count": max(1, min(10, _get_config_int("brave_count", "", 6))),
        "maximum_number_of_urls": max(1, min(10, _get_config_int("brave_max_urls", "", 5))),
        "maximum_number_of_tokens": max(1024, min(8192, _get_config_int("brave_max_tokens", "", 3072))),
        "context_threshold_mode": "balanced",
    }
    brave_url = _get_config_value("brave_context_url", "", "https://api.search.brave.com/res/v1/llm/context")
    req = urllib.request.Request(
        brave_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return None, f"联网检索服务返回 HTTP {exc.code}。{detail[:300]}".strip(), []
    except urllib.error.URLError as exc:
        return None, f"联网检索服务不可达：{exc.reason}", []
    except Exception as exc:
        return None, f"联网检索失败：{exc}", []

    grounding = data.get("grounding") if isinstance(data, dict) else {}
    sources = data.get("sources") if isinstance(data, dict) else {}
    generic = grounding.get("generic") if isinstance(grounding, dict) else []
    if not isinstance(generic, list):
        generic = []

    entries: list[str] = []
    source_items: list[dict[str, str]] = []
    max_urls = max(1, min(10, _get_config_int("brave_max_urls", "", 5)))
    for idx, item in enumerate(generic[:max_urls], start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or f"来源 {idx}").strip()
        snippets = item.get("snippets")
        if not isinstance(snippets, list):
            snippets = []
        meta = sources.get(url) if isinstance(sources, dict) else {}
        host = meta.get("hostname") if isinstance(meta, dict) else ""
        age = meta.get("age") if isinstance(meta, dict) else None
        age_text = ""
        if isinstance(age, list) and age:
            age_text = str(age[1] if len(age) > 1 else age[0]).strip()
        snippet_text = "\n".join(f"- {str(s).strip()}" for s in snippets[:4] if str(s).strip())
        if not snippet_text:
            continue
        header_bits = [f"[{idx}] {title}"]
        if host:
            header_bits.append(f"host={host}")
        if age_text:
            header_bits.append(f"date={age_text}")
        entry = "\n".join([
            " | ".join(header_bits),
            f"URL: {url}",
            snippet_text,
        ])
        entries.append(entry)
        plain_snippet = "\n".join(str(s).strip() for s in snippets[:3] if str(s).strip())
        if url:
            source_items.append(
                {
                    "title": title,
                    "url": url,
                    "host": host,
                    "published": age_text,
                    "snippet": plain_snippet,
                }
            )

    if not entries:
        return None, "联网检索已执行，但没有取回足够可用的网页上下文。", []

    context = "\n\n".join(entries)
    guidance = build_grounding_guidance(query, source_items, "实时网页")
    return f"{guidance}\n\nSearch query: {query}\n\n{context}", None, source_items


def build_grounded_payload(payload: dict[str, Any], context: str) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    grounded_messages = list(messages)
    system_message = {"role": "system", "content": context}

    if grounded_messages and isinstance(grounded_messages[-1], dict) and grounded_messages[-1].get("role") == "user":
        grounded_messages.insert(len(grounded_messages) - 1, system_message)
    else:
        grounded_messages.append(system_message)

    next_payload = dict(payload)
    next_payload["messages"] = grounded_messages
    return next_payload


@router.post("/chat")
async def chat_bridge(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    if not payload.get("web_search"):
        return relay_ollama(payload)

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return JSONResponse({"error": "invalid_messages"}, status_code=400)

    query = extract_search_query(messages)
    if not query:
        return synthetic_stream("联网模式已开启，但当前消息里没有可检索的文本问题。")

    context, error, _ = call_exa_search(query)
    if not context and not error:
        context, error, _ = call_brave_context(query)
    if not context and not error:
        error = "联网模式已开启，但服务器还没配置搜索 API key。请在 free_llm_config.json 中设置 exa_api_key 或 brave_api_key。"
    if error:
        return synthetic_stream(error)

    grounded_payload = build_grounded_payload(payload, context or "")
    return relay_ollama(grounded_payload)


# ── Chat file upload ─────────────────────────

CHAT_UPLOAD_DIR = Path(os.environ.get(
    "HONALA_CHAT_UPLOAD_DIR",
    "/var/www/html/chat/files",
)).resolve()
CHAT_UPLOAD_MAX_BYTES = int(os.environ.get("HONALA_CHAT_UPLOAD_MAX_BYTES", "20971520"))  # 20 MiB


@router.post("/chat-upload")
async def chat_upload(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse({"ok": False, "error": "缺少文件名"}, status_code=400)

    mime = (file.content_type or "application/octet-stream").lower()
    if not mime.startswith("image/"):
        return JSONResponse({"ok": False, "error": "仅支持图片上传"}, status_code=400)

    ext = os.path.splitext(file.filename)[1].lower()
    allowed = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif", ".svg"}
    if ext not in allowed:
        ext = ".jpg"

    CHAT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    filepath = CHAT_UPLOAD_DIR / filename

    size = 0
    try:
        contents = await file.read()
        if len(contents) > CHAT_UPLOAD_MAX_BYTES:
            return JSONResponse({"ok": False, "error": f"文件超过 {CHAT_UPLOAD_MAX_BYTES // 1048576}MB 限制"}, status_code=400)
        filepath.write_bytes(contents)
        size = len(contents)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"写入文件失败：{exc}"}, status_code=500)

    url_path = f"/files/{filename}"
    save_file_meta(filename, file.filename, mime, size, url_path)

    return JSONResponse({
        "ok": True,
        "url_path": url_path,
        "filename": filename,
        "original_name": file.filename,
        "mime_type": mime,
        "size": size,
    })


# ── Session & message persistence ─────────────

@router.get("/sessions")
async def api_list_sessions():
    return JSONResponse({"sessions": list_sessions()})


@router.post("/sessions")
async def api_create_or_update_session(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    sid = str(body.get("id") or "")
    if not sid:
        return JSONResponse({"error": "missing_id"}, status_code=400)

    name = str(body.get("name") or "")
    persona = str(body.get("persona") or "default")
    result = create_session(sid, name, persona)
    return JSONResponse({"ok": True, "session": result})


@router.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    deleted = delete_session(session_id)
    return JSONResponse({"ok": True, "deleted": deleted})


@router.get("/sessions/{session_id}/messages")
async def api_get_messages(session_id: str):
    msgs = get_messages(session_id)
    return JSONResponse({"messages": msgs, "session_id": session_id})


@router.post("/messages")
async def api_save_messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    sid = str(body.get("session_id") or "")
    if not sid:
        return JSONResponse({"error": "missing_session_id"}, status_code=400)

    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return JSONResponse({"error": "missing_messages"}, status_code=400)

    # Ensure session exists
    create_session(sid)

    # Replace strategy: delete old messages for this session, re-insert all
    delete_messages(sid)
    count = save_messages(sid, msgs)
    return JSONResponse({"ok": True, "count": count})


# ── User API provider management ──────────────

@router.get("/providers")
async def api_list_providers():
    return JSONResponse({"providers": list_providers()})


@router.post("/providers")
async def api_save_providers(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    providers_raw = body.get("providers") if isinstance(body, dict) else None
    if not isinstance(providers_raw, list):
        return JSONResponse({"error": "missing_providers"}, status_code=400)
    # Validate and sanitize
    clean = []
    for p in providers_raw:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        name = str(p.get("name") or "").strip()
        base_url = str(p.get("baseUrl") or "").strip()
        api_key = str(p.get("apiKey") or "").strip()
        if not pid or not base_url:
            continue
        clean.append({"id": pid, "name": name, "baseUrl": base_url, "apiKey": api_key})
    save_providers(clean)
    return JSONResponse({"ok": True, "count": len(clean)})


# ── External API proxy ────────────────────────

# ── Gemini LLM (via WARP SOCKS5 proxy) ────────

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

@router.post("/gemini-llm")
async def gemini_llm(request: Request):
    """Proxies streaming chat to Gemini API via local WARP SOCKS5 proxy."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    model = str(payload.get("model") or "gemini-3.5-flash").strip()
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    think = bool(payload.get("think", False))
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.9))

    api_key = _get_config_value("gemini_api_key", "")
    if not api_key:
        return JSONResponse({"error": "gemini_key_not_configured"}, status_code=503)

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            if contents:
                contents.insert(0, {"role": "user", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    if not contents:
        return JSONResponse({"error": "empty_contents"}, status_code=400)

    req_body = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": 8192,
        },
    }
    if think:
        pass

    url = f"{GEMINI_BASE}/models/{model}:streamGenerateContent?alt=sse&key={api_key}"

    try:
        import requests as req_lib
        upstream = req_lib.post(
            url,
            json=req_body,
            stream=True,
            proxies={"https": "socks5://127.0.0.1:40000"},
            timeout=120,
        )
        upstream.raise_for_status()
    except req_lib.exceptions.HTTPError as exc:
        detail = exc.response.text[:500] if exc.response else str(exc)
        return JSONResponse({"error": "gemini_upstream_error", "detail": detail}, status_code=exc.response.status_code if exc.response else 502)
    except req_lib.exceptions.ConnectionError as exc:
        return JSONResponse({"error": "gemini_unreachable", "detail": str(exc)}, status_code=502)
    except Exception as exc:
        return JSONResponse({"error": "gemini_error", "detail": str(exc)[:300]}, status_code=502)

    async def generate():
        try:
            for line in upstream.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if p.get("text"))
                if text:
                    yield make_stream_line(text)
        finally:
            upstream.close()
            yield make_stream_line("", done=True)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ── Image generation (DashScope) ──────────────

DASHSCOPE_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DASHSCOPE_VIDEO_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"


def _html_to_text(raw: str) -> str:
    """Very small HTML -> plain text extractor (no external deps)."""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|section|article|ul|ol|table)>", "\n", raw)
    text = re.sub(r"(?is)<[^>]+>", " ", raw)
    text = _htmlmod.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Tool registry (server-side execution) ─────
def _t_parse_args(a):
    if isinstance(a, dict):
        return a
    if not a:
        return {}
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _t_read_url(args):
    url = str(args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "（无效链接）"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; HonalaChat/1.0)",
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(2_000_000).decode("utf-8", errors="replace")
    except Exception as exc:
        return "（网页读取失败：%s）" % str(exc)[:100]
    if "html" in ctype or "<html" in raw[:500].lower():
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        text = _html_to_text(raw)
    else:
        title = ""
        text = raw
    return "[网页: %s]\n%s" % (title or url, text[:8000])


def _t_weather(args):
    loc = str(args.get("location") or "").strip()[:80]
    if not loc:
        return "（未提供地点）"
    try:
        geo_url = ("https://geocoding-api.open-meteo.com/v1/search?name="
                   + urllib.parse.quote(loc) + "&count=1&language=zh&format=json")
        with urllib.request.urlopen(geo_url, timeout=15) as r:
            geo = json.loads(r.read().decode("utf-8"))
        hits = geo.get("results") or []
        if not hits:
            return "（未找到地点：%s）" % loc
        g = hits[0]
        fc_url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
                  "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                  "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                  "&timezone=auto&forecast_days=3" % (g["latitude"], g["longitude"]))
        with urllib.request.urlopen(fc_url, timeout=15) as r:
            fc = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return "（天气查询失败：%s）" % str(exc)[:100]
    name = " ".join(x for x in [g.get("name"), g.get("admin1"), g.get("country")] if x)
    c = fc.get("current") or {}
    u = fc.get("current_units") or {}
    d = fc.get("daily") or {}
    out = ("[天气: %s]\n当前：%s%s（体感 %s%s），湿度 %s%s，风速 %s%s"
           % (name, c.get("temperature_2m"), u.get("temperature_2m", ""),
              c.get("apparent_temperature"), u.get("apparent_temperature", ""),
              c.get("relative_humidity_2m"), u.get("relative_humidity_2m", ""),
              c.get("wind_speed_10m"), u.get("wind_speed_10m", "")))
    if d.get("time"):
        out += "\n未来几天："
        for i in range(len(d["time"])):
            out += "\n%s %s~%s℃ 降水概率%s%%" % (
                d["time"][i], d["temperature_2m_min"][i], d["temperature_2m_max"][i],
                (d.get("precipitation_probability_max") or ["-"])[i])
    return out


def _t_get_time(args):
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone(timedelta(hours=8)))
    return "当前时间：%s（Asia/Shanghai）" % now.strftime("%Y-%m-%d %H:%M")


_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_CALC_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "pow": pow}
_CALC_CONSTS = {"pi": math.pi, "e": math.e}
# 把“书写体”数学符号归一化成 Python 认的写法
_CALC_SYMBOLS = {
    "\u00d7": "*", "\u00b7": "*", "\u2715": "*", "\u2716": "*", "\uff0a": "*",   # x · X X
    "\u00f7": "/", "\uff0f": "/",                                                      # /
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\uff0d": "-", "\uff0b": "+",    # - - - - +
    "\uff08": "(", "\uff09": ")",                                                      # ( )
    "\uff05": "%",                                                                       # %
    "^": "**",
    "\u00b2": "**2", "\u00b3": "**3",                                                  # ^2 ^3
    "\u03c0": "pi",                                                                      # pi
    "\u221a": "sqrt",                                                                    # sqrt
    "\uff0c": ",",                                                                       # ,
}


def _calc_norm(ex: str) -> str:
    ex = "".join(_CALC_SYMBOLS.get(ch, ch) for ch in ex)
    # √9 -> sqrt(9)（√ 后跟数字且没括号时自动补括号）
    ex = re.sub(r"\bsqrt\s*([0-9]+(?:\.[0-9]+)?)", r"sqrt(\1)", ex)
    return ex


def _calc_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("bad const")
    if isinstance(node, ast.BinOp):
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError("bad op")
        return op(_calc_eval(node.left), _calc_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError("bad op")
        return op(_calc_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CALC_CONSTS:
            return _CALC_CONSTS[node.id]
        raise ValueError("bad name")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCS and not node.keywords:
            return _CALC_FUNCS[node.func.id](*[_calc_eval(a) for a in node.args])
        raise ValueError("bad call")
    raise ValueError("bad node")


def _t_calculator(args):
    ex = str(args.get("expression") or "").strip()
    try:
        tree = ast.parse(_calc_norm(ex), mode="eval")
        val = _calc_eval(tree.body)
        if isinstance(val, (int, float)) and val == val and abs(val) != float("inf"):
            return "%s = %s" % (ex, val)
    except Exception:
        pass
    return "（计算失败）"


def _t_web_search(args):
    q = str(args.get("query") or "").strip()
    if not q:
        return "（搜索关键词为空，未执行）"
    context, error, items = call_exa_search(q)
    if not context and not error:
        context, error, items = call_brave_context(q)
    if error or not context:
        return "（搜索无结果）"
    return context


def _search_cards(args):
    try:
        q = str(args.get("query") or "").strip()
        context, error, items = call_exa_search(q)
        if not context and not error:
            context, error, items = call_brave_context(q)
        return [{"title": i.get("title", ""), "url": i.get("url", "")} for i in (items or [])[:6]]
    except Exception:
        return []


_TOOL_SCHEMA_SEARCH = {"type": "function", "function": {"name": "web_search", "description": "搜索实时网络信息。当需要最新新闻、事实核查、实时数据，或用户问题的答案超出你知识截止日期时使用。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索查询词，使用用户消息的语言"}}, "required": ["query"]}}}
_TOOL_SCHEMA_READ = {"type": "function", "function": {"name": "read_url", "description": "抓取指定网页的正文全文。当搜索结果只有摘要、需要网页正文或更多细节时使用。", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "要读取的完整网页地址（http/https）"}}, "required": ["url"]}}}
_TOOL_SCHEMA_WEATHER = {"type": "function", "function": {"name": "weather", "description": "查询某地当前天气与未来三天预报（结构化数据，比联网搜索更准）。", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市或地点名称，如 北京、Shanghai"}}, "required": ["location"]}}}
_TOOL_SCHEMA_TIME = {"type": "function", "function": {"name": "get_time", "description": "获取当前日期与时间。", "parameters": {"type": "object", "properties": {}}}}
_TOOL_SCHEMA_CALC = {"type": "function", "function": {"name": "calculator", "description": "进行精确数学计算。支持 + - * / ( ) 及幂（^ 或 **）、平方根 sqrt（或 √）、常量 pi/e；也接受 × ÷ − 等书写符号。", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如 (12.5*3+7)/2"}}, "required": ["expression"]}}}

def _t_generate_image(args):
    prompt = str(args.get("prompt") or "").strip()[:1000]
    if not prompt:
        return "（未提供图片描述）"
    api_key = _get_api_key("dashscope")
    if not api_key:
        return "（图片生成未配置 dashscope key）"
    size = str(args.get("size") or "1024*1024")
    req_body = {"model": "qwen-image-2.0-pro",
                "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
                "parameters": {"size": size}}
    req = urllib.request.Request(
        DASHSCOPE_IMAGE_URL, data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer %s" % api_key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as upstream:
            data = json.loads(upstream.read().decode("utf-8"))
    except Exception as exc:
        return "（图片生成失败：%s）" % str(exc)[:120]
    try:
        url = data["output"]["choices"][0]["message"]["content"][0]["image"]
    except (KeyError, IndexError, TypeError):
        url = ""
    if not url:
        return "（图片生成无返回）"
    return "[已生成图片] %s" % url


_TOOL_SCHEMA_IMAGE = {"type": "function", "function": {"name": "generate_image", "description": "根据文字描述生成图片。当用户要求画图、生成图片、配图时使用。", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "图片的详细描述"}, "size": {"type": "string", "description": "尺寸如 1024*1024（可选）"}}, "required": ["prompt"]}}}
TOOL_SCHEMAS = [_TOOL_SCHEMA_SEARCH, _TOOL_SCHEMA_READ, _TOOL_SCHEMA_WEATHER, _TOOL_SCHEMA_TIME, _TOOL_SCHEMA_CALC, _TOOL_SCHEMA_IMAGE]
_TOOL_FUNCS = {"web_search": _t_web_search, "read_url": _t_read_url, "weather": _t_weather, "get_time": _t_get_time, "calculator": _t_calculator, "generate_image": _t_generate_image}


def _tool_summary(name, args):
    if name == "web_search":
        return "联网搜索：" + str(args.get("query", ""))
    if name == "read_url":
        return "读取网页：" + str(args.get("url", ""))
    if name == "weather":
        return "天气：" + str(args.get("location", ""))
    if name == "get_time":
        return "获取当前时间"
    if name == "calculator":
        return "计算：" + str(args.get("expression", ""))
    if name == "generate_image":
        return "生成图片：" + str(args.get("prompt", ""))[:40]
    return name


def _exec_tool(name, args):
    fn = _TOOL_FUNCS.get(name)
    if not fn:
        return "（工具不可用：%s）" % name
    try:
        return fn(args)
    except Exception as exc:
        return "（工具执行出错：%s）" % str(exc)[:120]


def _resolve_agent_backend(provider_id, model):
    if provider_id:
        provs = list_providers()
        p = next((x for x in provs if x["id"] == provider_id), None)
        if not p:
            return None, None, False, True
        base = p["base_url"].strip().rstrip("/")
        return base + "/chat/completions", p["api_key"], False, True
    api_url, key, is_qwen = _get_backend(model)
    return api_url, key, is_qwen, False


def _agent_sanitize(messages):
    """剥离历史里的 reasoning_content，避免上游 400（DeepSeek 规定输入不得含它）。"""
    out = []
    for m in messages:
        if isinstance(m, dict) and "reasoning_content" in m:
            m = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(m)
    return out


def _agent_stream_call(url, key, model, messages, tools, is_qwen, external, temperature, top_p, think=False):
    """流式请求上游，逐块 yield：{"content"/"reasoning"/"tool_calls_delta"/"usage"/"finish_reason"}"""
    body = {"model": model, "messages": messages, "stream": True,
            "temperature": temperature, "top_p": top_p}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if not external:
        if is_qwen:
            body["enable_thinking"] = bool(think)
        elif not think:
            body["thinking"] = {"type": "disabled"}
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream",
                 "Authorization": "Bearer %s" % key}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP %s: %s" % (exc.code, detail[:300]))
    try:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            ds = line[5:].strip()
            if ds == "[DONE]":
                break
            try:
                obj = json.loads(ds)
            except Exception:
                continue
            u = obj.get("usage")
            if u:
                yield {"usage": u}
            choices = obj.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            fr = ch.get("finish_reason")
            if fr:
                yield {"finish_reason": fr}
            delta = ch.get("delta") or {}
            if delta.get("reasoning_content"):
                yield {"reasoning": delta["reasoning_content"]}
            if delta.get("content"):
                yield {"content": delta["content"]}
            if delta.get("tool_calls"):
                yield {"tool_calls_delta": delta["tool_calls"]}
    finally:
        resp.close()


def _merge_tool_deltas(acc, deltas):
    for d in deltas:
        idx = d.get("index", 0)
        tc = acc.setdefault(idx, {"id": None, "type": "function",
                                  "function": {"name": "", "arguments": ""}})
        if d.get("id"):
            tc["id"] = d["id"]
        if d.get("type"):
            tc["type"] = d["type"]
        fn = d.get("function") or {}
        if fn.get("name"):
            tc["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            tc["function"]["arguments"] += fn["arguments"]


def _agent_events(messages, model, provider_id, max_steps, temperature, top_p, think=False, agent_id=""):
    url, key, is_qwen, external = _resolve_agent_backend(provider_id, model)
    if not key:
        yield {"type": "error", "message": "api_key_not_configured"}
        return
    msgs = _agent_sanitize([dict(m) for m in messages])
    new_msgs = []
    usage = {}
    finish_reason = ""
    gen_ms = 0
    answered = False
    try:
        for step in range(max(1, max_steps)):
            if agent_id and _AGENT_STOP.get(agent_id):
                yield {"type": "done", "model": model, "usage": usage, "gen_ms": gen_ms,
                       "finish_reason": "stopped", "new_messages": new_msgs}
                return
            yield {"type": "status", "phase": "deciding", "step": step}
            acc = {}
            t0 = time.time()
            try:
                for ev in _agent_stream_call(url, key, model, msgs, TOOL_SCHEMAS, is_qwen, external, temperature, top_p, think):
                    if "usage" in ev:
                        usage = ev["usage"]
                    if "finish_reason" in ev:
                        finish_reason = ev["finish_reason"]
                    if "reasoning" in ev and think:
                        yield {"type": "thinking", "text": ev["reasoning"]}
                    if "content" in ev:
                        yield {"type": "answer_delta", "text": ev["content"]}
                    if "tool_calls_delta" in ev:
                        _merge_tool_deltas(acc, ev["tool_calls_delta"])
            except Exception as exc:
                yield {"type": "error", "message": "upstream_error: %s" % str(exc)[:200]}
                return
            if not acc:
                answered = True
                gen_ms = int((time.time() - t0) * 1000)
                break
            calls = [acc[i] for i in sorted(acc.keys())]
            asst = {"role": "assistant", "content": "", "tool_calls": calls}
            msgs.append(asst)
            new_msgs.append(asst)
            for tc in calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args = _t_parse_args(fn.get("arguments"))
                yield {"type": "tool_call", "id": tc.get("id"), "name": name,
                       "summary": _tool_summary(name, args)}
                content = _exec_tool(name, args)
                tev = {"type": "tool_result", "id": tc.get("id"), "name": name,
                       "ok": not str(content).startswith("（"), "content": str(content)[:2000]}
                if name == "web_search":
                    tev["cards"] = _search_cards(args)
                if name == "generate_image" and str(content).startswith("[已生成图片] "):
                    tev["image"] = str(content)[len("[已生成图片] "):].strip()
                yield tev
                tmsg = {"role": "tool", "tool_call_id": tc.get("id"), "content": content}
                msgs.append(tmsg)
                new_msgs.append(tmsg)
        if not answered:
            yield {"type": "status", "phase": "answering"}
            t0 = time.time()
            try:
                for ev in _agent_stream_call(url, key, model, msgs, None, is_qwen, external, temperature, top_p, think):
                    if "usage" in ev:
                        usage = ev["usage"]
                    if "finish_reason" in ev:
                        finish_reason = ev["finish_reason"]
                    if "reasoning" in ev and think:
                        yield {"type": "thinking", "text": ev["reasoning"]}
                    if "content" in ev:
                        yield {"type": "answer_delta", "text": ev["content"]}
            except Exception:
                pass
            gen_ms = int((time.time() - t0) * 1000)
    except Exception as exc:
        yield {"type": "error", "message": str(exc)[:200]}
        return
    yield {"type": "done", "model": model, "usage": usage, "gen_ms": gen_ms,
           "finish_reason": finish_reason, "new_messages": new_msgs}


_AGENT_STOP = {}


@router.post("/agent/abort")
async def agent_abort(request: Request):
    """前端点“停止”时调用，通知服务端终止对应 Agent 循环。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    aid = str((body or {}).get("id") or "")
    if aid:
        _AGENT_STOP[aid] = True
    return JSONResponse({"ok": True})


@router.post("/agent")
async def agent_endpoint(request: Request):
    """Server-side agent loop: decides, executes tools, streams events as NDJSON."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    model = str(body.get("model") or "deepseek-flash")
    provider_id = str(body.get("provider_id") or "") or None
    max_steps = int(body.get("max_steps") or 4)
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    think = bool(body.get("think", False))
    agent_id = request.headers.get("x-agent-id") or ""

    def generate():
        try:
            for ev in _agent_events(messages, model, provider_id, max_steps, temperature, top_p, think, agent_id):
                yield (json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8")
        finally:
            if agent_id:
                _AGENT_STOP.pop(agent_id, None)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/image-gen")
async def image_gen(request: Request):
    """Generate image via DashScope multimodal generation."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    prompt = str(body.get("prompt") or "").strip()[:1000]
    if not prompt:
        return JSONResponse({"error": "missing_prompt"}, status_code=400)
    api_key = _get_api_key("dashscope")
    if not api_key:
        return JSONResponse({"error": "dashscope_key_not_configured"}, status_code=503)
    size = str(body.get("size") or "1024*1024")
    req_body = {"model": "qwen-image-2.0-pro", "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]}, "parameters": {"size": size}}
    req = urllib.request.Request(DASHSCOPE_IMAGE_URL,
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as upstream:
            data = json.loads(upstream.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse({"error": "image_gen_failed", "detail": body[:300]}, status_code=exc.code)
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "image_gen_unreachable", "detail": str(exc.reason)}, status_code=502)
    try:
        url = data["output"]["choices"][0]["message"]["content"][0]["image"]
    except (KeyError, IndexError, TypeError):
        url = ""
    return JSONResponse({"ok": True, "url": url, "status": "done"})


# ── Video generation (DashScope) ──────────────

@router.post("/video-gen")
async def video_gen(request: Request):
    """Generate video via DashScope. Returns task_id for polling."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    prompt = str(body.get("prompt") or "").strip()[:1000]
    if not prompt:
        return JSONResponse({"error": "missing_prompt"}, status_code=400)
    api_key = _get_api_key("dashscope")
    if not api_key:
        return JSONResponse({"error": "dashscope_key_not_configured"}, status_code=503)
    req_body = {"model": "happyhorse-1.0-i2v", "input": {"prompt": prompt}, "parameters": {"duration": 5}}
    req = urllib.request.Request(DASHSCOPE_VIDEO_URL,
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "X-DashScope-Async": "enable"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as upstream:
            data = json.loads(upstream.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse({"error": "video_gen_failed", "detail": body[:300]}, status_code=exc.code)
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "video_gen_unreachable", "detail": str(exc.reason)}, status_code=502)
    task_id = data.get("output", {}).get("task_id", "")
    return JSONResponse({"ok": True, "task_id": task_id, "status": "pending"})


@router.get("/video-gen/{task_id}")
async def video_gen_status(task_id: str):
    """Poll video generation status from DashScope."""
    api_key = _get_api_key("dashscope")
    if not api_key:
        return JSONResponse({"error": "dashscope_key_not_configured"}, status_code=503)
    url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as upstream:
            data = json.loads(upstream.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse({"error": "poll_failed", "detail": body[:300]}, status_code=exc.code)
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "poll_unreachable", "detail": str(exc.reason)}, status_code=502)
    status = data.get("output", {}).get("task_status", "")
    url_out = ""
    if status == "SUCCEEDED":
        url_out = data.get("output", {}).get("video_url", "")
    return JSONResponse({"status": status, "url": url_out})
