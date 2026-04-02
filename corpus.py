"""
Corpus Management for RLM MCP Server

Handles loading and indexing of large text corpora (BYOKB - Bring Your Own Knowledge Base)
"""
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional

from config import CORPORA, CORPORA_DIR, auto_discover_corpora, register_corpus


class CorpusManager:
    """Manage multiple text corpora"""

    def __init__(self):
        self.corpora: Dict[str, Path] = {}
        self._register_default_corpora()

    def _register_default_corpora(self):
        """Register default corpus paths"""
        # Auto-discover corpora from state directory
        discovered = auto_discover_corpora()
        for name, path in discovered.items():
            self.corpora[name] = Path(path)

        # Register any pre-configured corpora
        for name, path in CORPORA.items():
            p = Path(path).expanduser()
            if p.exists():
                self.corpora[name] = p

    def register(self, name: str, path: str | Path) -> bool:
        """Register a new corpus"""
        p = Path(path).expanduser().resolve()
        if p.exists():
            self.corpora[name] = p
            # Update global CORPORA dict
            register_corpus(name, p)
            return True
        return False

    def list_available(self) -> List[Dict[str, str]]:
        """List all available corpora"""
        result = []
        for name, path in self.corpora.items():
            result.append({
                "name": name,
                "path": str(path),
                "type": "directory" if path.is_dir() else "file",
            })
        return result

    def get_path(self, name: str) -> Optional[Path]:
        """Get path for a named corpus"""
        return self.corpora.get(name)

    def exists(self, name: str) -> bool:
        """Check if corpus exists"""
        return name in self.corpora


class MarcosCorpus:
    """Marcos Lopez de Prado corpus loader (for directory of text files)"""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).expanduser().resolve()
        self.documents: Dict[str, str] = {}

    def load(self) -> Dict[str, str]:
        """Load all .txt documents from directory"""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Corpus path not found: {self.base_path}")

        txt_files = list(glob.glob(str(self.base_path / "*.txt")))

        for file_path in txt_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self.documents[filename] = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        self.documents[filename] = f.read()
                except Exception as e:
                    self.documents[filename] = f"Error reading file: {e}"

        return self.documents

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, any]]:
        """Simple keyword search across documents"""
        query_lower = query.lower()
        results = []

        for filename, text in self.documents.items():
            if query_lower in text.lower():
                # Find all occurrences
                start = 0
                while True:
                    idx = text.lower().find(query_lower, start)
                    if idx == -1:
                        break
                    snippet_start = max(0, idx - 300)
                    snippet_end = min(len(text), idx + 500)
                    results.append({
                        "filename": filename,
                        "match": text[idx:idx+len(query)],
                        "snippet": text[snippet_start:snippet_end],
                        "position": idx,
                    })
                    start = idx + 1
                    if len(results) >= max_results:
                        break
                if len(results) >= max_results:
                    break

        return results

    def get_document_names(self) -> List[str]:
        """Get list of all document filenames"""
        return list(self.documents.keys())

    def get_document(self, filename: str) -> Optional[str]:
        """Get a specific document by filename"""
        return self.documents.get(filename)

    def stats(self) -> Dict[str, any]:
        """Get corpus statistics"""
        total_chars = sum(len(doc) for doc in self.documents.values())
        return {
            "num_documents": len(self.documents),
            "total_characters": total_chars,
            "total_words": sum(len(doc.split()) for doc in self.documents.values()),
            "base_path": str(self.base_path),
        }


class SingleFileCorpus:
    """Single file corpus (e.g., knowledge_base.txt)"""

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path).expanduser().resolve()
        self.content: str = ""

    def load(self) -> str:
        """Load file content"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                self.content = f.read()
        except UnicodeDecodeError:
            with self.file_path.open("r", encoding="latin-1") as f:
                self.content = f.read()

        return self.content

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, any]]:
        """Search for query in content"""
        query_lower = query.lower()
        results = []
        start = 0

        while len(results) < max_results:
            idx = self.content.lower().find(query_lower, start)
            if idx == -1:
                break

            snippet_start = max(0, idx - 300)
            snippet_end = min(len(self.content), idx + 500)
            results.append({
                "filename": str(self.file_path.name),
                "match": self.content[idx:idx+len(query)],
                "snippet": self.content[snippet_start:snippet_end],
                "position": idx,
            })
            start = idx + 1

        return results

    def stats(self) -> Dict[str, any]:
        """Get corpus statistics"""
        return {
            "num_documents": 1,
            "total_characters": len(self.content),
            "total_words": len(self.content.split()),
            "file_path": str(self.file_path),
        }
