import re
from dataclasses import dataclass

from . import config

_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    heading: str | None = None
    page: int | None = None


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Break a paragraph that exceeds max_words into sentence groups (or word runs)."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _SENTENCE_END.split(paragraph):
        words = sentence.split()
        if len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            for i in range(0, len(words), max_words):
                pieces.append(" ".join(words[i : i + max_words]))
            continue
        if count + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    page: int | None = None,
    max_words: int = config.CHUNK_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split text into overlapping chunks, keeping paragraphs whole where possible.

    Markdown headings start a new chunk and are remembered as the chunk's heading.
    """
    chunks: list[Chunk] = []
    heading: str | None = None
    units: list[str] = []

    def flush(carry_overlap: bool) -> None:
        nonlocal units
        if not units:
            return
        chunks.append(Chunk(text="\n\n".join(units), heading=heading, page=page))
        tail: list[str] = []
        if carry_overlap:
            budget = overlap_words
            for unit in reversed(units):
                n = _word_count(unit)
                if n > budget:
                    break
                tail.insert(0, unit)
                budget -= n
        units = tail

    carried = 0  # how many units at the start of `units` are overlap from the last chunk
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        match = _HEADING.match(lines[0])
        if match:
            if len(units) > carried:
                flush(carry_overlap=False)
            units, carried = [], 0
            heading = match.group(1)
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        parts = _split_long(block, max_words) if _word_count(block) > max_words else [block]
        for part in parts:
            if units and sum(map(_word_count, units)) + _word_count(part) > max_words:
                if len(units) > carried:
                    flush(carry_overlap=True)
                    carried = len(units)
                else:
                    units, carried = [], 0
            units.append(part)
    if len(units) > carried:
        flush(carry_overlap=False)
    return chunks
