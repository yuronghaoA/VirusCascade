from __future__ import annotations

import random
import re
from pathlib import Path
from string import Template
from typing import Iterable, List


def fill_template(template: str, **kwargs) -> str:
    return Template(template).safe_substitute(**kwargs)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def strip_leading_list_prefix(text: str) -> str:
    cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*])\s+", "", text.strip())
    return cleaned


def normalize_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def clamp_words(text: str, max_words: int) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def split_clauses(text: str) -> List[str]:
    parts = re.split(r"[.;:!?]\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def recombine(a: str, b: str) -> str:
    a_parts = split_clauses(a)
    b_parts = split_clauses(b)
    if not a_parts or not b_parts:
        return f"{a} {b}".strip()
    cut_a = random.randint(1, len(a_parts))
    cut_b = random.randint(1, len(b_parts))
    combined = a_parts[:cut_a] + b_parts[cut_b:]
    if not combined:
        combined = a_parts + b_parts
    return " ".join(combined).strip()


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part and part.strip())
