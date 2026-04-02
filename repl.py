"""
Persistent REPL State Management for RLM

Adapted from claude_code_RLM/.claude/skills/rlm/scripts/rlm_repl.py
"""
from __future__ import annotations

import io
import os
import pickle
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import (
    DEFAULT_STATE_DIR,
    DEFAULT_STATE_FILE,
    DEFAULT_MAX_OUTPUT_CHARS,
)


class RlmReplError(RuntimeError):
    """RLM REPL exception"""
    pass


class RlmRepl:
    """Persistent REPL for RLM workflows"""

    def __init__(self, state_path: Path = DEFAULT_STATE_FILE):
        self.state_path = Path(state_path)
        self.state: Dict[str, Any] = {}
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        """Ensure state directory exists"""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        """Load state from pickle file"""
        if not self.state_path.exists():
            raise RlmReplError(
                f"No state found at {self.state_path}. "
                f"Initialize with rlm_init first."
            )
        with self.state_path.open("rb") as f:
            state = pickle.load(f)
        if not isinstance(state, dict):
            raise RlmReplError(f"Corrupt state file: {self.state_path}")
        return state

    def _save_state(self, state: Dict[str, Any]):
        """Save state to pickle file"""
        self._ensure_state_dir()
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.state_path)

    @staticmethod
    def _read_text_file(path: Path, max_bytes: int | None = None) -> str:
        """Read text file with fallback encoding"""
        if not path.exists():
            raise RlmReplError(f"Context file does not exist: {path}")
        data: bytes
        with path.open("rb") as f:
            data = f.read() if max_bytes is None else f.read(max_bytes)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    @staticmethod
    def _truncate(s: str, max_chars: int) -> str:
        """Truncate string to max chars"""
        if max_chars <= 0:
            return ""
        if len(s) <= max_chars:
            return s
        return s[:max_chars] + f"\n... [truncated to {max_chars} chars] ...\n"

    @staticmethod
    def _is_pickleable(value: Any) -> bool:
        """Check if value is pickleable"""
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False

    @staticmethod
    def _filter_pickleable(d: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Filter dict to only pickleable values"""
        kept: Dict[str, Any] = {}
        dropped: List[str] = []
        for k, v in d.items():
            if RlmRepl._is_pickleable(v):
                kept[k] = v
            else:
                dropped.append(k)
        return kept, dropped

    def init(self, context_path: Path, max_bytes: int | None = None) -> Dict[str, Any]:
        """Initialize REPL state from context file"""
        content = self._read_text_file(context_path, max_bytes)
        state: Dict[str, Any] = {
            "version": 1,
            "context": {
                "path": str(context_path),
                "loaded_at": time.time(),
                "content": content,
            },
            "buffers": [],
            "globals": {},
        }
        self._save_state(state)
        self.state = state

        return {
            "state_path": str(self.state_path),
            "context_path": str(context_path),
            "content_length": len(content),
            "loaded_at": state["context"]["loaded_at"],
        }

    def status(self, show_vars: bool = False) -> Dict[str, Any]:
        """Get current REPL status"""
        state = self._load_state()
        ctx = state.get("context", {})
        content = ctx.get("content", "")
        buffers = state.get("buffers", [])
        g = state.get("globals", {})

        result = {
            "state_path": str(self.state_path),
            "context_path": ctx.get("path"),
            "content_length": len(content),
            "num_buffers": len(buffers),
            "persisted_vars": len(g),
        }

        if show_vars and g:
            result["variables"] = sorted(g.keys())

        return result

    def reset(self) -> Dict[str, Any]:
        """Reset/delete REPL state"""
        if self.state_path.exists():
            self.state_path.unlink()
            return {"deleted": str(self.state_path), "status": "deleted"}
        return {"status": "no_state", "message": f"No state at {self.state_path}"}

    def peek(self, start: int = 0, end: int = 1000) -> str:
        """Peek at content range"""
        state = self._load_state()
        content = state.get("context", {}).get("content", "")
        return content[start:end]

    def grep(
        self,
        pattern: str,
        max_matches: int = 20,
        window: int = 120,
        flags: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search for pattern in content"""
        state = self._load_state()
        content = state.get("context", {}).get("content", "")
        out: List[Dict[str, Any]] = []

        for m in re.finditer(pattern, content, flags):
            start_idx, end_idx = m.span()
            snippet_start = max(0, start_idx - window)
            snippet_end = min(len(content), end_idx + window)
            out.append({
                "match": m.group(0),
                "span": (start_idx, end_idx),
                "snippet": content[snippet_start:snippet_end],
            })
            if len(out) >= max_matches:
                break

        return out

    def chunk_indices(self, size: int = 200_000, overlap: int = 0) -> List[Tuple[int, int]]:
        """Calculate chunk indices"""
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= size:
            raise ValueError("overlap must be < size")

        content_len = len(self.peek(0, float('inf'))) if hasattr(float('inf'), '__len__') else len(self._load_state().get("context", {}).get("content", ""))
        content = self._load_state().get("context", {}).get("content", "")
        n = len(content)
        spans: List[Tuple[int, int]] = []
        step = size - overlap
        for start in range(0, n, step):
            end = min(n, start + size)
            spans.append((start, end))
            if end >= n:
                break
        return spans

    def write_chunks(
        self,
        out_dir: str | os.PathLike,
        size: int = 200_000,
        overlap: int = 0,
        prefix: str = "chunk",
        encoding: str = "utf-8",
    ) -> List[str]:
        """Write chunks to files"""
        state = self._load_state()
        content = state.get("context", {}).get("content", "")
        spans = self.chunk_indices(size=size, overlap=overlap)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []

        for i, (s, e) in enumerate(spans):
            p = out_path / f"{prefix}_{i:04d}.txt"
            p.write_text(content[s:e], encoding=encoding)
            paths.append(str(p))

        return paths

    def add_buffer(self, text: str) -> None:
        """Add text to buffers"""
        state = self._load_state()
        buffers = state.setdefault("buffers", [])
        if not isinstance(buffers, list):
            buffers = []
            state["buffers"] = buffers
        buffers.append(str(text))
        self._save_state(state)

    def get_buffers(self) -> List[str]:
        """Get all buffers"""
        state = self._load_state()
        return state.get("buffers", [])

    def export_buffers(self, out_path: Path) -> Dict[str, Any]:
        """Export buffers to file"""
        state = self._load_state()
        buffers = state.get("buffers", [])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n\n".join(str(b) for b in buffers), encoding="utf-8")
        return {
            "output_path": str(out_path),
            "num_buffers": len(buffers),
        }

    def exec_code(
        self,
        code: str,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        warn_unpickleable: bool = False,
    ) -> Dict[str, Any]:
        """Execute Python code in REPL context"""
        state = self._load_state()

        ctx = state.get("context")
        if not isinstance(ctx, dict) or "content" not in ctx:
            raise RlmReplError("State is missing a valid 'context'. Re-run init.")

        buffers = state.setdefault("buffers", [])
        if not isinstance(buffers, list):
            buffers = []
            state["buffers"] = buffers

        persisted = state.setdefault("globals", {})
        if not isinstance(persisted, dict):
            persisted = {}
            state["globals"] = persisted

        # Build execution environment
        env: Dict[str, Any] = dict(persisted)
        env["context"] = ctx
        env["content"] = ctx.get("content", "")
        env["buffers"] = buffers

        # Add helper functions
        env["peek"] = lambda start=0, end=1000: ctx.get("content", "")[start:end]
        env["grep"] = lambda pat, max=20, win=120, fl=0: self.grep(pat, max, win, fl)
        env["chunk_indices"] = lambda size=200000, ov=0: self.chunk_indices(size, ov)
        env["write_chunks"] = lambda out, size=200000, ov=0, pref="chunk": self.write_chunks(out, size, ov, pref)
        env["add_buffer"] = lambda txt: buffers.append(str(txt))

        # Capture output
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, env, env)
        except Exception:
            traceback.print_exc(file=stderr_buf)

        # Update state from mutated environment
        maybe_ctx = env.get("context")
        if isinstance(maybe_ctx, dict) and "content" in maybe_ctx:
            state["context"] = maybe_ctx
            ctx = maybe_ctx

        maybe_buffers = env.get("buffers")
        if isinstance(maybe_buffers, list):
            state["buffers"] = maybe_buffers
            buffers = maybe_buffers

        # Persist new variables
        injected_keys = {
            "__builtins__", "context", "content", "buffers",
            "peek", "grep", "chunk_indices", "write_chunks", "add_buffer",
        }
        to_persist = {k: v for k, v in env.items() if k not in injected_keys}
        filtered, dropped = self._filter_pickleable(to_persist)
        state["globals"] = filtered

        self._save_state(state)

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()

        if dropped and warn_unpickleable:
            msg = "Dropped unpickleable variables: " + ", ".join(dropped)
            err = (err + ("\n" if err else "") + msg + "\n")

        return {
            "stdout": self._truncate(out, max_output_chars),
            "stderr": self._truncate(err, max_output_chars),
            "dropped_vars": dropped if dropped else [],
        }
