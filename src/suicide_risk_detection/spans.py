"""Offset-preserving text units and evidence candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"\S+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")
CLAUSE_BOUNDARY_RE = re.compile(r"\s*(?:;|(?<!\d),(?!\d)|—|–)\s*")


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    text: str
    kind: str

    @property
    def token_count(self) -> int:
        return len(TOKEN_RE.findall(self.text))


def _trim_span(text: str, start: int, end: int, kind: str) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return Span(start=start, end=end, text=text[start:end], kind=kind)


def sentence_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        candidate = _trim_span(text, start, match.start(), "sentence")
        if candidate:
            spans.append(candidate)
        start = match.end()
    candidate = _trim_span(text, start, len(text), "sentence")
    if candidate:
        spans.append(candidate)
    return spans or ([Span(0, len(text), text, "sentence")] if text else [])


def clause_spans(text: str, sentences: list[Span] | None = None) -> list[Span]:
    clauses: list[Span] = []
    for sentence in sentences or sentence_spans(text):
        local_start = 0
        for match in CLAUSE_BOUNDARY_RE.finditer(sentence.text):
            candidate = _trim_span(
                text, sentence.start + local_start, sentence.start + match.start(), "clause"
            )
            if candidate:
                clauses.append(candidate)
            local_start = match.end()
        candidate = _trim_span(text, sentence.start + local_start, sentence.end, "clause")
        if candidate:
            clauses.append(candidate)
    return clauses


def token_window_spans(text: str, window_sizes: list[int]) -> list[Span]:
    tokens = list(TOKEN_RE.finditer(text))
    spans: list[Span] = []
    for size in sorted(set(window_sizes)):
        if size <= 0 or len(tokens) < size:
            continue
        stride = max(1, size // 2)
        starts = list(range(0, len(tokens) - size + 1, stride))
        last = len(tokens) - size
        if not starts or starts[-1] != last:
            starts.append(last)
        for index in starts:
            start = tokens[index].start()
            end = tokens[index + size - 1].end()
            spans.append(Span(start, end, text[start:end], f"window_{size}"))
    return spans


def evidence_candidates(
    text: str, *, window_sizes: list[int], max_tokens: int
) -> list[Span]:
    sentences = sentence_spans(text)
    candidates = sentences + clause_spans(text, sentences) + token_window_spans(text, window_sizes)
    unique: dict[tuple[int, int], Span] = {}
    for span in candidates:
        if (
            0 < span.token_count <= max_tokens
            and ";" not in span.text
            and text[span.start : span.end] == span.text
        ):
            unique.setdefault((span.start, span.end), span)
    return sorted(unique.values(), key=lambda item: (item.start, item.end))
