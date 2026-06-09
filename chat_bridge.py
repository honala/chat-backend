# ADR-001 (2026-05-19): 图片上传"URL 方案"已废弃，回退纯 base64
# 三个失败节点：
#   1. Cloudflare Tunnel 路由 — 境外 edge 节点无法回源到新增 /api/chat-upload，
#      国内正常，路径注册问题
#   2. LLM 取图 — DeepSeek/DashScope 从 Cloudflare 取 ECS 文件失败，网络拓扑不通
#   3. 后端代理转 base64 — resolve_file_urls() 把磁盘文件读出转 base64 再发 LLM，
#      等于绕一圈没绕，方案自我否定
# 结论：单用户场景 base64 够用，不要再次尝试 URL 上传方案。
import json
import os
import uuid
import urllib.error
import urllib.request
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


@router.post("/search")
async def search_endpoint(request: Request):
    """Return structured search results as JSON for frontend-driven web search."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    query = str(body.get("query") or "").strip()[:500]
    if not query:
        return JSONResponse({"error": "missing_query"}, status_code=400)

    context, error, source_items = call_exa_search(query)
    provider = "exa"
    if not context and not error:
        context, error, source_items = call_brave_context(query)
        provider = "brave"
    if not context and not error:
        provider = "none"

    if error:
        return JSONResponse({"error": error, "results": [], "query": query}, status_code=502)

    # Use structured source_items directly — no need to parse the context string
    results: list[dict[str, str]] = []
    for item in source_items[:6]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "host": item.get("host", ""),
            "snippet": item.get("snippet", ""),
        })

    return JSONResponse({
        "results": results,
        "query": query,
        "provider": provider,
    })


@router.post("/free-llm")
async def free_llm(request: Request):
    """Proxy to LLM API using server-side key. Routes to DeepSeek or DashScope
    based on model prefix. Accepts Ollama-style payload, forwards to
    OpenAI-compatible endpoint, converts SSE stream back to NDJSON."""

    try:
        payload_value = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(payload_value, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    model = str(payload_value.get("model") or "deepseek-v4-flash")
    api_url, api_key, is_qwen = _get_backend(model)
    if not api_key:
        return JSONResponse({"error": "api_key_not_configured", "detail": f"服务器未配置 {model} 的 API 密钥"}, status_code=503)

    messages = payload_value.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    temperature = float(payload_value.get("options", {}).get("temperature", payload_value.get("temperature", 0.7)))
    top_p = float(payload_value.get("options", {}).get("top_p", payload_value.get("top_p", 0.9)))
    think_mode = bool(payload_value.get("think", False))

    req_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
    }
    if is_qwen:
        req_body["enable_thinking"] = think_mode
    elif think_mode:
        req_body["thinking"] = {"type": "enabled"}
    else:
        req_body["thinking"] = {"type": "disabled"}

    req = urllib.request.Request(
        api_url,
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse(
            {"error": "upstream_error", "status": exc.code, "detail": body[:500] or exc.reason},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "upstream_unreachable", "detail": str(exc.reason)}, status_code=502)

    def generate():
        net_buf = ""
        try:
            while True:
                chunk = upstream.readline()
                if not chunk:
                    break
                net_buf += chunk.decode("utf-8", errors="replace")
                while "\n" in net_buf:
                    line, net_buf = net_buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield make_stream_line("", done=True, model=model)
                        return
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices")
                        if not isinstance(choices, list) or not choices: continue
                        delta = choices[0].get("delta", {})
                        reasoning = str(delta.get("reasoning_content") or "")
                        content = str(delta.get("content") or "")
                        if reasoning:
                            line_data = {
                                "model": model,
                                "created_at": utc_iso(),
                                "message": {"role": "assistant", "content": "", "thinking": reasoning},
                                "done": False,
                            }
                            yield (json.dumps(line_data, ensure_ascii=False) + "\n").encode("utf-8")
                        if content:
                            yield make_stream_line(content, done=False, model=model)
                    except json.JSONDecodeError:
                        continue
        finally:
            upstream.close()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


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

@router.post("/external/chat")
async def external_chat_proxy(request: Request):
    """Proxies streaming chat to a user-configured external API.
    API keys stay on the server, never exposed to the browser."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    provider_id = str(payload.get("provider_id") or "")
    providers = list_providers()
    provider = next((p for p in providers if p["id"] == provider_id), None)
    if not provider:
        return JSONResponse({"error": "provider_not_found"}, status_code=404)

    base_url = provider["base_url"].strip().rstrip("/")
    api_key = provider["api_key"]
    model = str(payload.get("model") or "")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    temperature = float(payload.get("temperature", 0.7))
    top_p = float(payload.get("top_p", 0.9))

    req_body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse(
            {"error": "upstream_error", "status": exc.code, "detail": body[:500] or exc.reason},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "upstream_unreachable", "detail": str(exc.reason)}, status_code=502)

    def generate():
        net_buf = ""
        try:
            while True:
                chunk = upstream.readline()
                if not chunk:
                    break
                net_buf += chunk.decode("utf-8", errors="replace")
                while "\n" in net_buf:
                    line, net_buf = net_buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        yield make_stream_line("", done=True, model=model)
                        return
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices")
                        if not isinstance(choices, list) or not choices: continue
                        delta = choices[0].get("delta", {})
                        reasoning = str(delta.get("reasoning_content") or "")
                        content = str(delta.get("content") or "")
                        if reasoning:
                            meta = {"model": model, "created_at": utc_iso(),
                                    "message": {"role": "assistant", "content": "", "thinking": reasoning},
                                    "done": False}
                            yield (json.dumps(meta, ensure_ascii=False) + "\n").encode("utf-8")
                        if content:
                            yield make_stream_line(content, done=False, model=model)
                    except json.JSONDecodeError:
                        continue
        finally:
            upstream.close()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/external/tool")
async def external_tool_proxy(request: Request):
    """Non-streaming tool-calling proxy to user-configured external API."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    provider_id = str(payload.get("provider_id") or "")
    providers = list_providers()
    provider = next((p for p in providers if p["id"] == provider_id), None)
    if not provider:
        return JSONResponse({"error": "provider_not_found"}, status_code=404)

    base_url = provider["base_url"].strip().rstrip("/")
    api_key = provider["api_key"]
    model = str(payload.get("model") or "")
    messages = payload.get("messages")
    tools = payload.get("tools")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    req_body = {
        "model": model,
        "messages": messages,
        "tools": tools if isinstance(tools, list) else [],
        "tool_choice": "auto",
        "stream": False,
        "temperature": float(payload.get("temperature", 0.7)),
        "top_p": float(payload.get("top_p", 0.9)),
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as upstream:
            data = json.loads(upstream.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse({"error": "upstream_error", "status": exc.code, "detail": body[:500]}, status_code=exc.code)
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "upstream_unreachable", "detail": str(exc.reason)}, status_code=502)

    return JSONResponse(data)


@router.post("/free-llm/tool")
async def free_llm_tool(request: Request):
    """Non-streaming endpoint for tool calling. Model decides whether to search
    and returns tool_calls (or a direct answer). Frontend executes the search
    and feeds results back via the normal /free-llm streaming endpoint."""

    try:
        payload_value = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(payload_value, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)

    model = str(payload_value.get("model") or "deepseek-v4-flash")
    api_url, api_key, is_qwen = _get_backend(model)
    if not api_key:
        return JSONResponse({"error": "api_key_not_configured", "detail": f"服务器未配置 {model} 的 API 密钥"}, status_code=503)

    messages = payload_value.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse({"error": "invalid_messages"}, status_code=400)
    tools = payload_value.get("tools")
    if not isinstance(tools, list) or not tools:
        return JSONResponse({"error": "missing_tools", "detail": "tool calling 请求必须包含 tools 声明"}, status_code=400)

    temperature = float(payload_value.get("temperature", 0.7))
    top_p = float(payload_value.get("top_p", 0.9))

    req_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
    if is_qwen:
        req_body["enable_thinking"] = False
    else:
        req_body["thinking"] = {"type": "disabled"}

    req = urllib.request.Request(
        api_url,
        data=json.dumps(req_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        upstream = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return JSONResponse(
            {"error": "deepseek_upstream_error", "status": exc.code, "detail": body[:500] or exc.reason},
            status_code=exc.code,
        )
    except urllib.error.URLError as exc:
        return JSONResponse({"error": "deepseek_unreachable", "detail": str(exc.reason)}, status_code=502)

    try:
        data = json.loads(upstream.read().decode("utf-8"))
    finally:
        upstream.close()

    return JSONResponse(data)


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
