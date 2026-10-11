"""文本分块器 — 支持固定大小分块、句子边界感知、可配置 overlap"""
import re
from typing import Any


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "",
) -> list[dict[str, Any]]:
    """
    将长文本切分为多个块，返回 [{chunk_index, text, metadata}] 列表。
    - chunk_size: 每块最大字符数
    - chunk_overlap: 块间重叠字符数
    - separator: 优先按分隔符切分，"" 表示自动按自然段落/句子切分
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    if separator:
        segments = _split_by_separator(text, separator, chunk_size)
    else:
        segments = _split_by_natural_boundary(text, chunk_size)

    chunks = _merge_segments_with_overlap(segments, chunk_size, chunk_overlap)
    return [
        {"chunk_index": i, "text": c, "metadata": {"char_start": 0, "char_end": len(c)}}
        for i, c in enumerate(chunks)
    ]


def _split_by_natural_boundary(text: str, chunk_size: int) -> list[str]:
    """按双换行(段落)优先，再按单句切分"""
    paragraphs = text.split("\n\n")
    segments = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            segments.append(para)
        else:
            sentences = _split_sentences(para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) <= chunk_size:
                    segments.append(sent)
                else:
                    for i in range(0, len(sent), chunk_size // 2):
                        sub = sent[i:i + chunk_size].strip()
                        if sub:
                            segments.append(sub)
    return segments


def _split_sentences(text: str) -> list[str]:
    """中英文混合分句"""
    pattern = re.compile(
        r'(?<=[。！？.!?\n])\s*|'
        r'(?<=[；;])\s*'
    )
    parts = pattern.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_by_separator(text: str, sep: str, chunk_size: int) -> list[str]:
    parts = text.split(sep)
    return [p.strip() for p in parts if p.strip()]


def _merge_segments_with_overlap(segments: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = ""
    for seg in segments:
        if not current:
            current = seg
        elif len(current) + len(seg) + 1 <= chunk_size:
            current = current + "\n" + seg
        else:
            chunks.append(current.strip())
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n" + seg
            else:
                current = seg
    if current.strip():
        chunks.append(current.strip())
    return chunks
