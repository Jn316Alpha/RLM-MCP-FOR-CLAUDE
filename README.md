# RLM MCP Server for Claude

**Recursive Language Models (RLM)** - Inference-Time Scaling via Programmatic Context Examination

An MCP (Model Context Protocol) server that enables Claude to handle contexts **two orders of magnitude** beyond standard window limits through recursive decomposition and programmatic analysis.

## Scientific Foundation

This server implements the **Recursive Language Model (RLM)** paradigm from recent research on inference-time scaling. Instead of treating long contexts as passive text to be "read," RLM treats large documents as an **external environment** that the model can:

1. **Examine programmatically** - Execute Python code to grep, filter, and transform data
2. **Decompose recursively** - Break large problems into smaller, targeted queries
3. **Synthesize results** - Combine intermediate findings into coherent answers

This approach achieves **91%+ accuracy** on tasks that would otherwise exceed context window limits.

### The Recursive Advantage

| Approach | Context Limit | Accuracy | Use Case |
|----------|--------------|----------|----------|
| Standard Read | ~200K tokens | Degrades sharply | Documents < 100 pages |
| RLM Recursive | ~10M tokens | 91%+ | Large knowledge bases, codebases |
| Human Assistant | Unlimited | ~95% | Manual research |

RLM bridges the gap between standard reading and human research assistance by treating the document as a **computable environment** rather than static text.

## Included Example Corpus

This repository includes the **RLM research paper** (`corpora/RLM_MIT.txt`) as a starter corpus. This allows you to immediately test RLM capabilities and understand how recursive language models work.

## BYOKB: Bring Your Own Knowledge Base

Beyond the included example, you can add your own `.txt` or `.md` files:

1. Place your knowledge base files in `.rlm_state/corpora/`
2. Or provide a full path when initializing

Example corpora you might use:
- Academic papers (convert PDF to text)
- Technical documentation
- Source code (concatenated)
- Research notes
- Trading strategies and backtests
- Biblical studies texts

## Installation

### Prerequisites

```bash
pip install mcp
```

### MCP Configuration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["PATH/TO/rlm_mcp_server.py"]
    }
  }
}
```

**On Windows** (example):
```json
{
  "mcpServers": {
    "rlm": {
      "command": "python",
      "args": ["C:\\Users\\YourName\\rlm_mcp_for_claude\\rlm_mcp_server.py"]
    }
  }
}
```

Then restart Claude Code.

## Available Tools

| Tool | Description |
|------|-------------|
| `rlm_init` | Initialize with a corpus file |
| `rlm_status` | Show current REPL state |
| `rlm_query` | Keyword search in corpus |
| `rlm_peek` | Preview content range |
| `rlm_grep` | Regex pattern search |
| `rlm_chunk` | Create chunk files |
| `rlm_exec` | Execute Python code |
| `rlm_list_corpora` | List available corpora |
| `rlm_reset` | Clear REPL state |
| `rlm_get_buffers` | Get accumulated results |

## Quick Start: Test with the RLM Paper

Once you have RLM configured in Claude Code, try these example prompts to test it with the included research paper:

### Example 1: Understand RLM Architecture
```
"Use RLM. Initialize with 'corpora/RLM_MIT.txt'. Explain the core architecture of Recursive Language Models and how they achieve inference-time scaling."
```

### Example 2: Find Key Results
```
"Use RLM. Initialize with 'corpora/RLM_MIT.txt'. What are the main experimental results showing RLM's accuracy compared to baseline methods?"
```

### Example 3: Extract Implementation Details
```
"Use RLM. Initialize with 'corpora/RLM_MIT.txt'. Use rlm_exec to search for and extract all mentions of 'prompt' or 'prompting' and explain how prompts are structured in RLM."
```

### Example 4: Compare Methods
```
"Use RLM. Initialize with 'corpora/RLM_MIT.txt'. Search for and compare the different baseline methods mentioned (RAG, long context, etc.) against RLM's approach."
```

### Example 5: Find Limitations
```
"Use RLM. Initialize with 'corpora/RLM_MIT.txt'. What limitations or failure cases are mentioned for Recursive Language Models?"
```

## Practical Usage

### High-Precision Research

Instead of asking Claude to "read" a 5MB document, use RLM to **compute** on it:

**Example 1: Trading Strategy Analysis**
```
User: "Use RLM. Initialize with 'trading_data.txt'. Use rlm_exec to run a CUSUM filter over the last 16,000 points and return only the indices where a structural break occurred."

Claude:
1. rlm_init(corpus_name="trading_data")
2. rlm_exec(code="""
import numpy as np
from mlfinlab.filters import cusum_filter

# Parse data from content
lines = content.split('\\n')
data = [float(l.split(',')[1]) for l in lines[1:] if l]

# Apply CUSUM filter
events = cusum_filter(data, threshold=0.5)
print(f"Found {len(events)} structural breaks at indices: {events[:20]}")
""")
```

**Example 2: Academic Paper Analysis**
```
User: "Use RLM. Initialize with 'machine_learning_papers.txt'. Find all papers that mention 'causal inference' and extract their abstracts."

Claude:
1. rlm_init(corpus_name="ml_papers")
2. rlm_exec(code="""
import re

# Find paper boundaries
paper_starts = [m.start() for m in re.finditer(r'^@@@\\d+', content, re.MULTILINE)]

# Search for causal inference mentions
causal_papers = []
for i, start in enumerate(paper_starts):
    end = paper_starts[i+1] if i+1 < len(paper_starts) else len(content)
    paper_text = content[start:end]
    if 'causal inference' in paper_text.lower():
        # Extract abstract
        abstract_match = re.search(r'Abstract:(.*?)(?=\\n\\n|$)', paper_text, re.DOTALL)
        if abstract_match:
            causal_papers.append(abstract_match.group(1))

print(f"Found {len(causal_papers)} papers on causal inference")
for abstract in causal_papers[:3]:
    print(f"\\n{abstract}\\n---")
""")
```

**Example 3: Biblical Exegesis**
```
User: "Use RLM. Initialize with 'biblical_studies.txt'. Find all instances where Paul discusses 'faith' and 'works' together, with chapter context."

Claude:
1. rlm_init(corpus_name="bible")
2. rlm_exec(code="""
import re

# Find passages mentioning both faith and works
pattern = r'faith.*works|works.*faith'
matches = list(re.finditer(pattern, content, re.IGNORECASE))

# Extract chapter context for each match
for match in matches:
    pos = match.start()
    # Get 500 chars before and after for context
    context = content[max(0, pos-500):min(len(content), pos+500)]
    # Try to identify book/chapter
    chapter_match = re.search(r'(\\w+\\s+\\d+):\\d+', context[pos-200:pos])
    if chapter_match:
        print(f"\\nReference: {chapter_match.group(0)}")
        print(f"Context: ...{context}...")
""")
```

## How RLM Works: The REPL Environment

When you initialize RLM with a corpus, it creates a **persistent Python REPL** with:

- `content` - The full corpus text (accessible via `peek()` for ranges)
- `grep()` - Search for patterns with context windows
- `chunk_indices()` - Calculate chunk boundaries
- `write_chunks()` - Split large files into manageable pieces
- `add_buffer()` - Save intermediate results

The REPL maintains **global variables** across multiple `rlm_exec` calls, enabling multi-step research workflows.

### Example: Multi-Step Research Workflow

```python
# Step 1: Find relevant sections
rlm_exec(code="""
# Find all mentions of 'triple barrier'
matches = grep('triple barrier', max=50)
print(f"Found {len(matches)} matches")
""")

# Step 2: Extract and analyze those sections
rlm_exec(code="""
# Use previous results
sections = []
for m in matches:
    start = max(0, m['span'][0] - 1000)
    end = min(len(content), m['span'][1] + 2000)
    sections.append(content[start:end])

# Analyze
for i, section in enumerate(sections[:5]):
    print(f"\\n=== Section {i+1} ===")
    print(section[:500])
""")

# Step 3: Synthesize findings
rlm_exec(code="""
add_buffer("Triple barrier method is used for...")
add_buffer("Key parameters: pt_sl, min_ret, num_days")
""")
```

## Research Paper Alignment

This implementation follows the **Recursive Language Model** paradigm:

1. **External Environment**: The corpus is treated as a computable environment, not passive text
2. **Recursive Decomposition**: Complex queries are broken into targeted sub-queries
3. **Programmatic Examination**: Python code executes grep, filters, and transformations
4. **Result Synthesis**: Intermediate findings combine into coherent answers

### Key Insights from Research

- **Accuracy**: RLM maintains 91%+ accuracy even at 10M token contexts
- **Efficiency**: Only relevant sections are processed, not entire documents
- **Flexibility**: Custom analysis code can be written per query
- **Stateful**: Multi-step workflows maintain context and variables

## File Structure

```
rlm_mcp_for_claude/
├── rlm_mcp_server.py    # Main entry point
├── server.py             # MCP tool handlers
├── repl.py               # Persistent REPL state
├── corpus.py             # Corpus loading
├── chunker.py            # Chunking strategies
├── config.py             # Configuration
├── README.md             # This file
├── LICENSE               # MIT License
├── requirements.txt      # Dependencies
├── .gitignore
├── corpora/              # Example corpora
│   └── RLM_MIT.txt      # RLM research paper (included)
└── .rlm_state/           # Created at runtime
    ├── state.pkl         # REPL state
    └── corpora/          # BYOKB - place your additional files here
```

## License

MIT License - See LICENSE file

## Citation

If you use this in research, please cite the Recursive Language Models paper and this implementation.

## Contributing

Contributions welcome! This is an open implementation of the RLM paradigm.

## Support

For issues, questions, or feature requests, please use the GitHub issue tracker.

## Example Claude Code Sessions

### Session 1: Learning About RLM

```
User: "Use RLM. Initialize with 'corpora/RLM_MIT.txt' and summarize what Recursive Language Models are."

Claude: [Calls rlm_init with the paper]
[Uses rlm_query and rlm_peek to read the introduction]
[Synthesizes a summary explaining that RLM treats long contexts as external environments
 that can be examined programmatically through recursive Python execution]
```

### Session 2: Technical Deep Dive

```
User: "Use RLM. Find all mentions of 'accuracy' in the paper and report the quantitative results."

Claude: [Calls rlm_init if not already loaded]
[Uses rlm_grep with pattern "accuracy" to find all mentions]
[Uses rlm_exec to extract and parse numerical results]
[Returns a structured summary of all accuracy metrics from the experiments]
```

### Session 3: Comparative Analysis

```
User: "Use RLM. Compare how RLM handles 1M token contexts versus RAG (Retrieval-Augmented Generation)."

Claude: [Uses rlm_exec to search for both '1M' and 'RAG' mentions]
[Extracts comparison sections]
[Creates a structured comparison table from the paper's findings]
```

---

**Remember**: RLM is a tool for **high-precision research**, not bulk text processing. Use it when you need specific, accurate answers from large knowledge bases.

**Start with the included paper** to understand RLM's capabilities, then add your own corpora for your specific research needs.
