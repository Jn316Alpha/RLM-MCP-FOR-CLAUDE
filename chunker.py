"""
Chunking Strategies for RLM MCP Server

Provides various methods to chunk large text into processable pieces
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class Chunker:
    """Base chunker class"""

    def __init__(self, content: str):
        self.content = content
        self.length = len(content)

    def chunk_by_size(
        self,
        size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Tuple[int, int, str]]:
        """Chunk by character size with optional overlap"""
        if size <= 0:
            raise ValueError("size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= size:
            raise ValueError("overlap must be < size")

        chunks = []
        step = size - overlap

        for start in range(0, self.length, step):
            end = min(self.length, start + size)
            chunk_text = self.content[start:end]
            chunks.append((start, end, chunk_text))
            if end >= self.length:
                break

        return chunks

    def chunk_by_lines(
        self,
        lines_per_chunk: int = 1000,
        overlap_lines: int = 0,
    ) -> List[Tuple[int, int, str]]:
        """Chunk by line count"""
        lines = self.content.splitlines(keepends=True)
        chunks = []
        step = lines_per_chunk - overlap_lines

        for i in range(0, len(lines), step):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_text = "".join(chunk_lines)
            start_pos = self.content.find(chunk_text)
            chunks.append((start_pos, start_pos + len(chunk_text), chunk_text))
            if end >= len(lines):
                break

        return chunks

    def chunk_by_markdown(
        self,
        max_size: int = DEFAULT_CHUNK_SIZE,
    ) -> List[Tuple[int, int, str]]:
        """Chunk by markdown headings (semantic chunking)"""
        chunks = []

        # Find all heading positions
        heading_pattern = r'^#{1,6}\s+.*$'
        lines = self.content.splitlines(keepends=True)

        heading_positions = []
        current_pos = 0

        for i, line in enumerate(lines):
            if re.match(heading_pattern, line):
                heading_positions.append((i, current_pos))
            current_pos += len(line)

        # Create chunks between headings
        for i, (line_idx, pos) in enumerate(heading_positions):
            start = pos
            if i + 1 < len(heading_positions):
                end = heading_positions[i + 1][1]
            else:
                end = self.length

            chunk_text = self.content[start:end]

            # Split if too large
            if len(chunk_text) > max_size:
                sub_chunks = self._split_large_chunk(chunk_text, max_size)
                for sub in sub_chunks:
                    chunks.append((start, start + len(sub), sub))
                    start += len(sub)
            else:
                chunks.append((start, end, chunk_text))

        return chunks

    def chunk_by_json_objects(self, max_size: int = DEFAULT_CHUNK_SIZE) -> List[Tuple[int, int, str]]:
        """Chunk JSON array into individual objects"""
        chunks = []

        try:
            import json
            data = json.loads(self.content)

            if isinstance(data, list):
                for i, obj in enumerate(data):
                    chunk_text = json.dumps(obj, indent=2)
                    start = self.content.find(chunk_text)
                    if start != -1:
                        chunks.append((start, start + len(chunk_text), chunk_text))
            else:
                # Single JSON object
                chunks.append((0, self.length, self.content))

        except json.JSONDecodeError:
            # Not valid JSON, fall back to size-based chunking
            return self.chunk_by_size(max_size, 0)

        return chunks

    def chunk_by_sentences(
        self,
        max_size: int = DEFAULT_CHUNK_SIZE,
    ) -> List[Tuple[int, int, str]]:
        """Chunk by sentences (approximate)"""
        # Simple sentence boundary detection
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, self.content)

        chunks = []
        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append((current_start, current_start + len(current_chunk), current_chunk))
                    current_start += len(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append((current_start, current_start + len(current_chunk), current_chunk))

        return chunks

    def _split_large_chunk(
        self,
        text: str,
        max_size: int,
    ) -> List[str]:
        """Split a chunk that's too large"""
        if len(text) <= max_size:
            return [text]

        parts = []
        remaining = text

        while len(remaining) > max_size:
            split_pos = max_size
            # Try to split at whitespace
            while split_pos > 0 and not remaining[split_pos].isspace():
                split_pos -= 1
            if split_pos == 0:
                split_pos = max_size

            parts.append(remaining[:split_pos])
            remaining = remaining[split_pos:].lstrip()

        if remaining:
            parts.append(remaining)

        return parts

    def write_chunks_to_files(
        self,
        chunks: List[Tuple[int, int, str]],
        output_dir: Path,
        prefix: str = "chunk",
    ) -> List[str]:
        """Write chunks to individual files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, (start, end, text) in enumerate(chunks):
            chunk_path = output_dir / f"{prefix}_{i:04d}.txt"
            chunk_path.write_text(text, encoding="utf-8")
            paths.append(str(chunk_path))

        return paths


class SemanticChunker(Chunker):
    """Semantic chunking using content patterns"""

    def __init__(self, content: str):
        super().__init__(content)

    def chunk_by_log_entries(
        self,
        max_entries: int = 1000,
    ) -> List[Tuple[int, int, str]]:
        """Chunk log file by entries (timestamp patterns)"""
        # Common log timestamp patterns
        log_patterns = [
            r'^\d{4}-\d{2}-\d{2}[T ]',  # ISO date
            r'^\d{2}/\d{2}/\d{4}',      # MM/DD/YYYY
            r'^\w{3}\s+\d{1,2}\s+',     # Mon DD HH:MM:SS
            r'^\[\d{2}/\w{3}/\d{4}',   # [01/Jan/2024
        ]

        lines = self.content.splitlines(keepends=True)
        chunk_starts = [0]

        for i, line in enumerate(lines):
            for pattern in log_patterns:
                if re.match(pattern, line):
                    if i - chunk_starts[-1] >= max_entries:
                        chunk_starts.append(i)
                    break

        chunks = []
        for i in range(len(chunk_starts)):
            start = chunk_starts[i]
            if i + 1 < len(chunk_starts):
                end = chunk_starts[i + 1]
            else:
                end = len(lines)
            chunk_text = "".join(lines[start:end])
            start_pos = self.content.find(chunk_text)
            chunks.append((start_pos, start_pos + len(chunk_text), chunk_text))

        return chunks

    def chunk_by_code_blocks(
        self,
        language: Optional[str] = None,
    ) -> List[Tuple[int, int, str]]:
        """Chunk markdown by code blocks"""
        code_block_pattern = r'```(\w*)\n(.*?)```'

        chunks = []
        last_end = 0

        for match in re.finditer(code_block_pattern, self.content, re.DOTALL):
            start, end = match.span()

            # Add text before code block
            if start > last_end:
                chunks.append((
                    last_end,
                    start,
                    self.content[last_end:start]
                ))

            # Add code block
            block_lang = match.group(1)
            if language is None or block_lang == language:
                chunks.append((start, end, match.group(0)))

            last_end = end

        # Add remaining text
        if last_end < self.length:
            chunks.append((last_end, self.length, self.content[last_end:]))

        return [c for c in chunks if c[2].strip()]
