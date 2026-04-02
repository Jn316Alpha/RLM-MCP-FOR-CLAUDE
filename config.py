"""
RLM MCP Server Configuration
"""
import os
from pathlib import Path

# Server info
SERVER_NAME = "rlm-mcp-server"
SERVER_VERSION = "1.0.0"

# Installation directory
INSTALL_DIR = Path(__file__).parent.resolve()

# State management
DEFAULT_STATE_DIR = INSTALL_DIR / ".rlm_state"
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "state.pkl"

# Chunking defaults
DEFAULT_CHUNK_SIZE = 200_000  # characters
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_MAX_OUTPUT_CHARS = 8_000

# Grep defaults
DEFAULT_MAX_MATCHES = 20
DEFAULT_WINDOW = 120

# BYOKB: Users must provide their own corpora
# Place your .txt or .md files in the state directory to use with rlm_init
CORPORA_DIR = DEFAULT_STATE_DIR / "corpora"

# Built-in example corpora (included with repository)
_BUILTIN_CORPORA = {
    "rlm_paper": str(INSTALL_DIR / "corpora" / "RLM_MIT.txt"),
}

# Available corpora (BYOKB - Bring Your Own Knowledge Base)
# Users should place their files in CORPORA_DIR
CORPORA = _BUILTIN_CORPORA.copy()

def register_corpus(name: str, path: str | Path) -> bool:
    """
    Register a corpus for use with RLM.

    Args:
        name: Name to reference the corpus
        path: Path to the corpus file or directory

    Returns:
        True if corpus exists and was registered
    """
    p = Path(path).expanduser().resolve()
    if p.exists():
        CORPORA[name] = str(p)
        return True
    return False

def auto_discover_corpora() -> dict:
    """
    Auto-discover corpora files in the CORPORA_DIR.

    Returns:
        Dictionary of corpus name -> path
    """
    discovered = {}
    if CORPORA_DIR.exists():
        for ext in ['*.txt', '*.md', '*.json']:
            for path in CORPORA_DIR.glob(ext):
                name = path.stem
                discovered[name] = str(path)
    return discovered
