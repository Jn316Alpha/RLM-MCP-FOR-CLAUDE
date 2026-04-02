#!/usr/bin/env python3
"""
RLM MCP Server - Standalone Entry Point

Main entry point for running the RLM MCP server via stdio.
Usage: python rlm_mcp_server.py

For Claude Code, add to config:
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["FULL_PATH_TO_THIS_FILE"]
    }
  }
}
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server import main

if __name__ == "__main__":
    asyncio.run(main())
