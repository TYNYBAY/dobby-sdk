# Examples

## Web Search Agent

A CLI-based streaming agent with web search capabilities.

### Setup

```bash
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...
```

Get your Tavily API key from [tavily.com](https://app.tavily.com/home).

### Usage

```bash
python examples/web_search_agent.py "What are the latest AI news?"
```

### Features

- **Streaming output** - Responses stream in real-time
- **Tool logging** - See when tools are called and their results
- **Color-coded CLI** - Easy to read tool calls and responses

### Example Output

```
Query: What are the latest AI developments?

──────────────────────────────────────────────────

Model: gpt-4o-mini

🔧 Tool Call: tavily_search
   Inputs: {'query': 'latest AI developments 2024'}

✓ Tool Result:
   [{'title': 'AI News...', 'url': 'https://...'}...]

Based on my search, here are the latest AI developments...

──────────────────────────────────────────────────
Tokens: 1234 (in: 456, out: 778)
```
