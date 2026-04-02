"""
RLM MCP Server - Main Server Implementation

Provides MCP tools for Recursive Language Model workflows.
"""
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from config import SERVER_NAME, SERVER_VERSION, CORPORA
from repl import RlmRepl, RlmReplError
from corpus import CorpusManager, MarcosCorpus, SingleFileCorpus
from chunker import Chunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(SERVER_NAME)

# Create MCP server
app = Server(SERVER_NAME)

# Global state
_repl_instance: RlmRepl = None
_corpus_manager = CorpusManager()


def get_repl() -> RlmRepl:
    """Get or create REPL instance"""
    global _repl_instance
    if _repl_instance is None:
        from config import DEFAULT_STATE_FILE
        _repl_instance = RlmRepl(DEFAULT_STATE_FILE)
    return _repl_instance


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available RLM resources (corpora)"""
    resources = []

    for name, path in CORPORA.items():
        p = Path(path)
        if p.exists():
            resources.append(Resource(
                uri=f"corpus://{name}",
                name=f"Corpus: {name}",
                description=f"Marcos Lopez de Prado corpus at {path}",
                mimeType="text/plain",
            ))

    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a corpus resource"""
    if uri.startswith("corpus://"):
        name = uri.replace("corpus://", "")
        path = _corpus_manager.get_path(name)
        if path and path.exists():
            if path.is_dir():
                # List files in directory
                files = list(path.glob("*.txt"))
                return json.dumps({
                    "name": name,
                    "path": str(path),
                    "type": "directory",
                    "files": [f.name for f in files[:50]],  # Limit to 50
                    "total_files": len(files),
                }, indent=2)
            else:
                # File info
                return json.dumps({
                    "name": name,
                    "path": str(path),
                    "type": "file",
                    "size": path.stat().st_size,
                }, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available RLM tools"""
    return [
        Tool(
            name="rlm_init",
            description="Initialize RLM REPL with a context file (corpus)",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_path": {
                        "type": "string",
                        "description": "Path to the context file (corpus) to load",
                    },
                    "corpus_name": {
                        "type": "string",
                        "description": "Optional: Name of pre-registered corpus (marcos, marcos_corpus, sinclair, biblical_studies)",
                        "enum": list(CORPORA.keys()),
                    },
                    "max_bytes": {
                        "type": "number",
                        "description": "Optional: Max bytes to read from file",
                    },
                },
            },
        ),
        Tool(
            name="rlm_status",
            description="Show current RLM REPL status (context, buffers, variables)",
            inputSchema={
                "type": "object",
                "properties": {
                    "show_vars": {
                        "type": "boolean",
                        "description": "Show persisted variable names",
                    },
                },
            },
        ),
        Tool(
            name="rlm_query",
            description="Search the corpus for a query string (keyword search)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum number of results (default: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="rlm_peek",
            description="Preview a range of characters from the loaded content",
            inputSchema={
                "type": "object",
                "properties": {
                    "start": {
                        "type": "number",
                        "description": "Start character position (default: 0)",
                    },
                    "end": {
                        "type": "number",
                        "description": "End character position (default: 1000)",
                    },
                },
            },
        ),
        Tool(
            name="rlm_grep",
            description="Search for a regex pattern in the content with context window",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for",
                    },
                    "max_matches": {
                        "type": "number",
                        "description": "Maximum matches (default: 20)",
                    },
                    "window": {
                        "type": "number",
                        "description": "Context window size in chars (default: 120)",
                    },
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="rlm_chunk",
            description="Create chunk files from the loaded content",
            inputSchema={
                "type": "object",
                "properties": {
                    "size": {
                        "type": "number",
                        "description": "Chunk size in characters (default: 200000)",
                    },
                    "overlap": {
                        "type": "number",
                        "description": "Overlap between chunks (default: 0)",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory for chunk files",
                    },
                },
            },
        ),
        Tool(
            name="rlm_exec",
            description="Execute Python code in the REPL context",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    },
                },
                "required": ["code"],
            },
        ),
        Tool(
            name="rlm_list_corpora",
            description="List all available corpora",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="rlm_reset",
            description="Reset/clear the REPL state",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="rlm_get_buffers",
            description="Get all buffers (accumulated results)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls"""

    try:
        if name == "rlm_init":
            return await _rlm_init(arguments)
        elif name == "rlm_status":
            return await _rlm_status(arguments)
        elif name == "rlm_query":
            return await _rlm_query(arguments)
        elif name == "rlm_peek":
            return await _rlm_peek(arguments)
        elif name == "rlm_grep":
            return await _rlm_grep(arguments)
        elif name == "rlm_chunk":
            return await _rlm_chunk(arguments)
        elif name == "rlm_exec":
            return await _rlm_exec(arguments)
        elif name == "rlm_list_corpora":
            return await _rlm_list_corpora(arguments)
        elif name == "rlm_reset":
            return await _rlm_reset(arguments)
        elif name == "rlm_get_buffers":
            return await _rlm_get_buffers(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    except RlmReplError as e:
        return [TextContent(type="text", text=f"RLM Error: {e}")]
    except Exception as e:
        logger.exception(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


# Tool implementations

async def _rlm_init(arguments: dict) -> list[TextContent]:
    """Initialize REPL with context"""
    repl = get_repl()

    context_path = arguments.get("context_path")
    corpus_name = arguments.get("corpus_name")
    max_bytes = arguments.get("max_bytes")

    # Determine context path
    if corpus_name:
        path = _corpus_manager.get_path(corpus_name)
        if path:
            if path.is_dir():
                # For directories, we need to handle differently
                # Use the Marcos corpus loader
                corpus = MarcosCorpus(path)
                docs = corpus.load()
                # Create a temp file with all content
                from config import DEFAULT_STATE_DIR
                DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
                temp_file = DEFAULT_STATE_DIR / f"{corpus_name}_combined.txt"
                combined = "\n\n" + "="*70 + "\n\n"
                combined = combined.join(docs.values())
                temp_file.write_text(combined, encoding="utf-8")
                context_path = str(temp_file)
            else:
                context_path = str(path)
        else:
            return [TextContent(type="text", text=f"Corpus '{corpus_name}' not found")]

    if not context_path:
        return [TextContent(type="text", text="Error: Must provide context_path or corpus_name")]

    result = repl.init(Path(context_path), max_bytes)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _rlm_status(arguments: dict) -> list[TextContent]:
    """Get REPL status"""
    repl = get_repl()
    show_vars = arguments.get("show_vars", False)
    result = repl.status(show_vars=show_vars)
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _rlm_query(arguments: dict) -> list[TextContent]:
    """Search corpus for query"""
    repl = get_repl()
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 10)

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    # Use grep to find matches
    results = repl.grep(query, max_matches=max_results, window=300)

    output = {
        "query": query,
        "num_matches": len(results),
        "matches": results,
    }

    return [TextContent(type="text", text=json.dumps(output, indent=2))]


async def _rlm_peek(arguments: dict) -> list[TextContent]:
    """Peek at content"""
    repl = get_repl()
    start = arguments.get("start", 0)
    end = arguments.get("end", 1000)

    content = repl.peek(start, end)

    return [TextContent(type="text", text=json.dumps({
        "start": start,
        "end": end,
        "content": content,
    }, indent=2))]


async def _rlm_grep(arguments: dict) -> list[TextContent]:
    """Grep for pattern"""
    repl = get_repl()
    pattern = arguments.get("pattern", "")
    max_matches = arguments.get("max_matches", 20)
    window = arguments.get("window", 120)

    if not pattern:
        return [TextContent(type="text", text="Error: pattern is required")]

    results = repl.grep(pattern, max_matches=max_matches, window=window)

    return [TextContent(type="text", text=json.dumps({
        "pattern": pattern,
        "num_matches": len(results),
        "matches": results,
    }, indent=2))]


async def _rlm_chunk(arguments: dict) -> list[TextContent]:
    """Create chunks"""
    repl = get_repl()
    size = arguments.get("size", 200_000)
    overlap = arguments.get("overlap", 0)
    output_dir = arguments.get("output_dir", ".claude/rlm_state/chunks")

    paths = repl.write_chunks(output_dir, size=size, overlap=overlap)

    return [TextContent(type="text", text=json.dumps({
        "chunk_size": size,
        "overlap": overlap,
        "num_chunks": len(paths),
        "output_dir": output_dir,
        "chunk_files": paths[:10],  # Show first 10
        "total_chunks": len(paths),
    }, indent=2))]


async def _rlm_exec(arguments: dict) -> list[TextContent]:
    """Execute Python code"""
    repl = get_repl()
    code = arguments.get("code", "")

    if not code:
        return [TextContent(type="text", text="Error: code is required")]

    result = repl.exec_code(code)

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _rlm_list_corpora(arguments: dict) -> list[TextContent]:
    """List available corpora"""
    corpora = _corpus_manager.list_available()

    return [TextContent(type="text", text=json.dumps({
        "num_corpora": len(corpora),
        "corpora": corpora,
    }, indent=2))]


async def _rlm_reset(arguments: dict) -> list[TextContent]:
    """Reset REPL state"""
    repl = get_repl()
    result = repl.reset()
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _rlm_get_buffers(arguments: dict) -> list[TextContent]:
    """Get all buffers"""
    repl = get_repl()
    buffers = repl.get_buffers()

    return [TextContent(type="text", text=json.dumps({
        "num_buffers": len(buffers),
        "buffers": buffers,
    }, indent=2))]


async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
