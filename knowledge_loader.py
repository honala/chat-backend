"""文档加载器 — 支持 PDF / Word / HTML / Markdown / TXT 格式"""
import os
from pathlib import Path
from typing import Optional


def load_document(file_path: str) -> Optional[str]:
    """根据文件扩展名自动选择解析器，返回纯文本内容"""
    fp = Path(file_path)
    if not fp.exists():
        return None
    ext = fp.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(fp)
    elif ext in (".docx", ".doc"):
        return _load_docx(fp)
    elif ext in (".htm", ".html"):
        return _load_html(fp)
    elif ext in (".md", ".markdown"):
        return _load_text(fp)
    elif ext == ".txt":
        return _load_text(fp)
    else:
        return None


def get_supported_extensions() -> list[str]:
    return [".pdf", ".docx", ".doc", ".html", ".htm", ".md", ".markdown", ".txt"]


def _load_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return fp.read_text(encoding="gbk")
        except Exception:
            return fp.read_text(encoding="utf-8", errors="replace")


def _load_pdf(fp: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(fp))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts) if parts else ""
    except ImportError:
        return _fallback_text(fp, "请安装 pypdf: pip install pypdf")
    except Exception as e:
        return _fallback_text(fp, f"PDF 解析失败: {e}")


def _load_docx(fp: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(fp))
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts) if parts else ""
    except ImportError:
        return _fallback_text(fp, "请安装 python-docx: pip install python-docx")
    except Exception as e:
        return _fallback_text(fp, f"Word 解析失败: {e}")


def _load_html(fp: Path) -> str:
    try:
        from bs4 import BeautifulSoup
        html = fp.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
    except ImportError:
        return _fallback_text(fp, "请安装 beautifulsoup4: pip install beautifulsoup4")
    except Exception as e:
        return _fallback_text(fp, f"HTML 解析失败: {e}")


def _fallback_text(fp: Path, msg: str) -> str:
    raw = _load_text(fp)
    return f"[{msg}]\n\n--- 原始内容 ---\n\n{raw}" if raw else f"[{msg}]"
