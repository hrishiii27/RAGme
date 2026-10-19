"""Phase 2 — Chunking 🧑‍💻 YOUR ASSIGNMENT

This is the core of Stage A of our experiments. You need to implement multiple chunking
strategies so we can benchmark them against each other.

The `Chunker` interface is simple: `chunk(doc: Document) -> list[Chunk]`.
Each chunk needs `chunk_id`, `doc_id`, `text`, `start_char`, and `end_char`.

Task:
Implement the chunkers below so they pass `uv run pytest tests/test_chunking.py`.
"""

from __future__ import annotations

import re

import tiktoken

from bisect import bisect_left, bisect_right
from rag.chunking.base import Chunker
from rag.types import Chunk, Document


def _make_chunk(doc: Document, strategy: str, index: int, text: str,
                start_char: int, end_char: int) -> Chunk:
    """Single place that builds a Chunk. Adjust the field names to match your Chunk."""
    return Chunk(
        doc_id=doc.doc_id,
        chunk_id=f"{doc.doc_id}::{strategy}::{index}",
        text=text,
        start_char=start_char,
        end_char=end_char,
    )
 
 
class FixedTokenChunker(Chunker):
    """Fixed-size token windows (tiktoken cl100k_base) with token overlap.
 
    start_char/end_char are exact: token spans are mapped back to character
    offsets via byte offsets, and chunk.text is always text[start_char:end_char].
    """
 
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding("cl100k_base")
 
    @property
    def strategy_name(self) -> str:
        return f"fixed-tokens-{self.chunk_size}-{self.overlap}"
 
    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []
 
        tokens = self.enc.encode(text, disallowed_special=())
 
        # ends[i] = byte offset (in the UTF-8 text) where token i ends.
        ends: list[int] = []
        total = 0
        for t in tokens:
            total += len(self.enc.decode_single_token_bytes(t))
            ends.append(total)
 
        # char_starts[j] = byte offset where character j starts.
        char_starts: list[int] = []
        pos = 0
        for ch in text:
            char_starts.append(pos)
            pos += len(ch.encode("utf-8"))
 
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        n = len(tokens)
        for a in range(0, n, step):
            b = min(a + self.chunk_size, n)
            start_byte = ends[a - 1] if a > 0 else 0
            end_byte = ends[b - 1]
            # A token boundary can fall inside a multi-byte character; include
            # the whole character at each edge so the slice is always valid.
            start_char = bisect_right(char_starts, start_byte) - 1
            end_char = bisect_left(char_starts, end_byte)
            chunks.append(_make_chunk(doc, self.strategy_name, len(chunks),
                                      text[start_char:end_char], start_char, end_char))
            if b == n:  # don't emit a tail chunk that is pure overlap
                break
        return chunks
 
 
class RecursiveCharacterChunker(Chunker):
    """Splits on ["\\n\\n", "\\n", ". ", " ", ""], keeping chunks <= chunk_size chars.
 
    Pass 1 splits the text into contiguous spans that each fit in chunk_size,
    trying the coarsest separator first and recursing into oversized pieces.
    Pass 2 greedily merges spans into chunks and carries up to `overlap`
    characters (whole spans) into the next chunk.
    """
 
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
 
    def __init__(self, chunk_size: int = 1500, overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap
 
    @property
    def strategy_name(self) -> str:
        return f"recursive-chars-{self.chunk_size}-{self.overlap}"
 
    def _split(self, text: str, start: int, end: int, level: int) -> list[tuple[int, int]]:
        if end - start <= self.chunk_size:
            return [(start, end)]
        sep = self.SEPARATORS[level]
        if sep == "":  # last resort: hard cut
            return [(i, min(i + self.chunk_size, end))
                    for i in range(start, end, self.chunk_size)]
        spans: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            idx = text.find(sep, pos, end)
            seg_end = end if idx == -1 else idx + len(sep)  # keep the separator
            spans.extend(self._split(text, pos, seg_end, level + 1))
            pos = seg_end
        return spans
 
    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []
        spans = self._split(text, 0, len(text), 0)
 
        chunks: list[Chunk] = []
 
        def emit(s: int, e: int) -> None:
            piece = text[s:e]
            if piece.strip():
                chunks.append(_make_chunk(doc, self.strategy_name, len(chunks), piece, s, e))
 
        window: list[tuple[int, int]] = []
        fresh = False  # does the window hold spans not yet emitted?
        for span in spans:
            while window and span[1] - window[0][0] > self.chunk_size:
                if fresh:
                    emit(window[0][0], window[-1][1])
                    fresh = False
                    # keep only the tail that fits within the overlap budget
                    while window and window[-1][1] - window[0][0] > self.overlap:
                        window.pop(0)
                else:
                    window.pop(0)  # only overlap left over; drop to make room
            window.append(span)
            fresh = True
        if fresh:
            emit(window[0][0], window[-1][1])
        return chunks
 
 
class MarkdownHeaderChunker(Chunker):
    """One chunk per header section (#, ##, ###), prefixed with the header path.
 
    chunk.text = "# A > ## B\\nbody"; start_char/end_char point at the section
    in the original document (header line through end of body).
    """
 
    HEADER_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
 
    @property
    def strategy_name(self) -> str:
        return "markdown-headers"
 
    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        if not text:
            return []
 
        sections: list[tuple[str | None, int, int, int]] = []  # crumb, start, body_start, end
        stack: list[tuple[int, str]] = []  # (level, "## Title")
        crumb: str | None = None
        sec_start = body_start = 0
        in_fence = False
        pos = 0
 
        for line in text.splitlines(keepends=True):
            bare = line.rstrip("\r\n")
            if bare.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
            m = None if in_fence else self.HEADER_RE.match(bare)
            if m:
                sections.append((crumb, sec_start, body_start, pos))
                level = len(m.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, f"{m.group(1)} {m.group(2)}"))
                crumb = " > ".join(h for _, h in stack)
                sec_start, body_start = pos, pos + len(line)
            pos += len(line)
        sections.append((crumb, sec_start, body_start, len(text)))
 
        chunks: list[Chunk] = []
        for crumb, s_start, b_start, s_end in sections:
            body = text[b_start:s_end]
            if not body.strip():
                continue  # header with no body of its own (children carry the path)
            if crumb:
                out, start_char = f"{crumb}\n{body.strip()}", s_start
            else:  # preamble before the first header
                out, start_char = body.strip(), b_start + (len(body) - len(body.lstrip()))
            end_char = b_start + len(body.rstrip())
            chunks.append(_make_chunk(doc, self.strategy_name, len(chunks), out, start_char, end_char))
        return chunks