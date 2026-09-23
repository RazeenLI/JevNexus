"""Small I/O helpers: atomic writes, JSON/JSONL, hashing."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    """Write ``text`` so that readers never observe a partially written file."""
    path = Path(path)
    ensure_parent(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | os.PathLike, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic short hash of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def safe_name(value: str) -> str:
    """Filesystem-safe name component (keeps it readable)."""
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep)
