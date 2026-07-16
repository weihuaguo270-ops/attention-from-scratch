"""Cross-platform console helpers for Windows GBK terminals."""

from __future__ import annotations

import sys
from typing import Any, Optional, TextIO

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
OK = "[OK]"

_EMOJI_MAP = {
    "✅": PASS,
    "❌": FAIL,
    "⚠️": WARN,
    "⚠": WARN,
    "🎉": OK,
}


def configure_stdio(*, errors: str = "replace") -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors=errors)
        except Exception:
            pass


def sanitize(text: str, *, encoding: Optional[str] = None) -> str:
    out = str(text)
    for src, dst in _EMOJI_MAP.items():
        out = out.replace(src, dst)
    enc = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        out.encode(enc)
        return out
    except UnicodeEncodeError:
        return out.encode(enc, errors="replace").decode(enc, errors="replace")


def safe_print(*args: Any, sep: str = " ", end: str = "\n",
               file: Optional[TextIO] = None, flush: bool = False) -> None:
    stream = file or sys.stdout
    text = sanitize(sep.join(str(a) for a in args), encoding=getattr(stream, "encoding", None))
    try:
        print(text, end=end, file=stream, flush=flush)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "ascii"
        fallback = text.encode(enc, errors="replace").decode(enc, errors="replace")
        print(fallback, end=end, file=stream, flush=flush)


configure_stdio()
