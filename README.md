# OpenRouter Free MCP

An MCP (Model Context Protocol) server that discovers and evaluates free AI models on OpenRouter.

[GitHub](https://github.com/TheOneAC/openrouter-free-mcp)

## Features

- **List all free models** — fetches real-time data from OpenRouter API
- **Top model recommendations** — scores models on Agent, Coding, and Size criteria
- **Vision companion** — automatically recommends a free vision-capable model when your chosen model doesn't support vision

## Tools

| Tool | Description |
|---|---|
| `list_free_models` | List all free models (price = $0) |
| `get_top_free_models` | Top N models by agent/coding/size/balanced score |
| `get_best_coding_model` | Best model for coding tasks |
| `get_best_agent_model` | Best model for agent/tool-calling |
| `generate_team_config` | Recommend the best free model for each agent team role (coordinator, coder, researcher, vision) |

## Scoring

Each model scored 0-100 across three dimensions:

- **Agent:** tools(+35), structured_outputs(+30), reasoning(+20), tool_choice(+10), parallel_tool_calls(+5)
- **Coding:** context_length(0-30), tools(+25), reasoning(+25), structured_outputs(+20)
- **Size:** parsed model parameter count (1T+=100, 400B+=90, ...)

## Usage

### One-Click Install

```bash
pip install mcp && curl -sL https://raw.githubusercontent.com/TheOneAC/openrouter-free-mcp/main/server.py -o openrouter-free-mcp-server.py
```

Then run:

```bash
python3 openrouter-free-mcp-server.py
```

### With Claude Code

Add to `claude.json`:

```json
{
  "mcpServers": {
    "free-token-mcp": {
      "command": "python3",
      "args": ["/path/to/openrouter-free-mcp-server.py"]
    }
  }
}
```

### With Hermes

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  free-token-mcp:
    command: python3
    args: ["/path/to/openrouter-free-mcp-server.py"]
    enabled: true
```

## Requirements

- Python 3.10+
- `mcp` package (`pip install mcp`)

## License

MIT
