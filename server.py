"""MCP server that discovers and evaluates free OpenRouter models.

Fetches model list from OpenRouter API (no API key needed for listing),
filters for free models (prompt=0, completion=0), evaluates Agent and Coding
capability based on supported_parameters, context_length, and model size.

Usage:
    python server.py
"""

import json
import os
import re
import subprocess
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    id: str
    name: str
    context_length: int
    supported_parameters: list[str]
    pricing: dict[str, str]
    description: str
    supports_vision: bool = False


# ── API client ───────────────────────────────────────────────────────────────

def fetch_models() -> list[dict[str, Any]]:
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"User-Agent": "free-token-mcp/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter API HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter API network error: {e.reason}")
    return data["data"]


# ── Model size parsing ───────────────────────────────────────────────────────

def parse_model_size(description: str) -> float:
    """Parse parameter size from model description.

    Returns total parameters in billions (e.g. 120 for "120B", 1000 for "1T").
    For MoE models, returns the largest total size (not the active params).
    """
    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(T|t)\b', description)
    if matches:
        nums = [float(n) * 1000 for n, _ in matches]
        return max(nums)

    matches = re.findall(r'(\d+(?:\.\d+)?)\s*(B|b)\b', description)
    if matches:
        nums = [float(n) for n, _ in matches]
        return max(nums)

    return 0.0


def format_size(billions: float) -> str:
    if billions >= 1000:
        return f"{billions / 1000:.1f}T"
    elif billions >= 1:
        return f"{billions:.0f}B"
    elif billions > 0:
        return f"{billions:.1f}B"
    return "unknown"


# ── Evaluation logic ─────────────────────────────────────────────────────────

def _get_pricing_value(pricing: Any, key: str) -> str:
    if isinstance(pricing, dict):
        return pricing.get(key, "0")
    if isinstance(pricing, list) and pricing:
        return pricing[0].get(key, "0")
    return "0"


def filter_free_models(models: list[dict]) -> list[ModelInfo]:
    result = []
    for m in models:
        pricing = m.get("pricing", {})
        if _get_pricing_value(pricing, "prompt") == "0" and _get_pricing_value(pricing, "completion") == "0":
            arch = m.get("architecture", {}) or {}
            input_mods = arch.get("input_modalities") or []
            result.append(ModelInfo(
                id=m["id"],
                name=m.get("name", m["id"]),
                context_length=m.get("context_length") or 0,
                supported_parameters=m.get("supported_parameters") or [],
                pricing=pricing if isinstance(pricing, dict) else {"prompt": "0", "completion": "0"},
                description=m.get("description", ""),
                supports_vision="image" in input_mods,
            ))
    return result


def _params(model: ModelInfo) -> set[str]:
    return set(model.supported_parameters)


def compute_agent_score(model: ModelInfo) -> float:
    p = _params(model)
    score = 0.0
    if "tools" in p: score += 35
    if "structured_outputs" in p: score += 30
    if "reasoning" in p: score += 20
    if "tool_choice" in p: score += 10
    if "parallel_tool_calls" in p: score += 5
    return score


def compute_coding_score(model: ModelInfo) -> float:
    ctx = model.context_length
    ctx_score = (30 if ctx >= 200_000 else 27 if ctx >= 128_000
                 else 22 if ctx >= 64_000 else 15 if ctx >= 32_000
                 else 8 if ctx >= 8_000 else 3)
    p = _params(model)
    tools_score = 25 if "tools" in p else 0
    reasoning_score = 25 if "reasoning" in p else 0
    structured_score = 20 if "structured_outputs" in p else 0
    return float(ctx_score + tools_score + reasoning_score + structured_score)

def compute_size_score(description: str) -> float:
    """Score based on model parameter count (larger = more capable)."""
    size_b = parse_model_size(description)
    if size_b >= 1000: return 100      # 1T+
    if size_b >= 400: return 90        # 400B+
    if size_b >= 100: return 75        # 100B+
    if size_b >= 30: return 55         # 30B+
    if size_b >= 10: return 35         # 10B+
    if size_b >= 1: return 15          # 1B+
    return 0                           # unknown


def _strengths(model: ModelInfo, size_b: float) -> list[str]:
    s = []
    p = _params(model)
    if "tools" in p: s.append("tool use")
    if "structured_outputs" in p: s.append("structured outputs")
    if "reasoning" in p: s.append("reasoning")
    if model.context_length >= 128_000:
        s.append(f"long context ({model.context_length:,} tokens)")
    if size_b >= 10:
        s.append(f"large model ({format_size(size_b)})")
    if model.supports_vision:
        s.append("vision (multimodal)")
    return s


# ── Vision companion ────────────────────────────────────────────────────────────

def find_best_vision_companion(raw_models: list[dict] | None = None) -> str | None:
    """Find the best free vision-capable model to use as a vision companion."""
    try:
        raw = raw_models if raw_models is not None else fetch_models()
        candidates = []
        for m in raw:
            mid = m["id"]
            if mid == "openrouter/free":
                continue
            arch = m.get("architecture", {}) or {}
            input_mods = arch.get("input_modalities") or []
            if "image" not in input_mods:
                continue
            pricing = m.get("pricing", {})
            if _get_pricing_value(pricing, "prompt") != "0":
                continue
            if _get_pricing_value(pricing, "completion") != "0":
                continue
            ctx = m.get("context_length") or 0
            candidates.append((ctx, mid))

        if not candidates:
            return "openrouter/free"
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    except Exception:
        return "openrouter/free"


# ── MCP server (FastMCP) ─────────────────────────────────────────────────────

mcp = FastMCP("free-token-mcp")


@mcp.tool()
def list_free_models() -> str:
    """List all free models on OpenRouter (price = $0 for both prompt and completion)"""
    try:
        raw = fetch_models()
        free = filter_free_models(raw)
        free.sort(key=lambda m: m.context_length, reverse=True)
        output = {
            "count": len(free),
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "context_length": m.context_length,
                    "pricing": m.pricing,
                    "supported_parameters": m.supported_parameters,
                    "model_size": format_size(parse_model_size(m.description)),
                    "supports_vision": m.supports_vision,
                }
                for m in free
            ],
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_best_coding_model() -> str:
    """Return the free model with the highest coding capability score."""
    return get_top_free_models(criterion="coding", top_n=1)


@mcp.tool()
def get_best_agent_model() -> str:
    """Return the free model with the highest agent and tool-calling capability score."""
    return get_top_free_models(criterion="agent", top_n=1)


@mcp.tool()
def launch_claude_code(model_id: str = "auto") -> str:
    """Pick the best free coding model and launch a new Claude Code session.

    Automatically detects OpenRouter API key from environment or Hermes auth pool,
    writes a temporary claude.json, and opens a new terminal with Claude Code.

    Args:
        model_id: Model ID to use (default "auto" = pick best free coding model)
    """
    try:
        # 1. Resolve model
        if model_id == "auto":
            raw = fetch_models()
            free = filter_free_models(raw)
            scored = []
            for m in free:
                if m.id == "openrouter/free":
                    continue
                scored.append((compute_coding_score(m), m.id, m.name))
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored:
                return json.dumps({"error": "No free models found."})
            model_id = scored[0][1]
            model_name = scored[0][2]
        else:
            model_name = model_id

        # 2. Get API key
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            try:
                result = subprocess.run(
                    ["hermes", "auth", "list", "--json"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    creds = json.loads(result.stdout)
                    for c in creds if isinstance(creds, list) else [creds]:
                        if "openrouter" in str(c).lower():
                            api_key = c.get("api_key", c.get("key", ""))
                            break
            except Exception:
                pass

        if not api_key:
            return json.dumps({
                "error": "No OpenRouter API key found. "
                         "Set OPENROUTER_API_KEY env var or add via: hermes auth add openrouter <key>"
            })

        # 3. Write temp claude.json
        tmp_dir = Path(tempfile.mkdtemp(prefix="free-token-mcp-"))
        config = {
            "model": model_id,
            "baseUrl": "https://openrouter.ai/api/v1",
            "mcpServers": {},
        }
        config_path = tmp_dir / "claude.json"
        config_path.write_text(json.dumps(config, indent=2))

        # 4. Launch Claude Code in new terminal
        script = (
            f'cd {tmp_dir} && '
            f'ANTHROPIC_BASE_URL=https://openrouter.ai/api/v1 '
            f'ANTHROPIC_API_KEY={api_key} '
            f'claude --model {model_id}'
        )
        subprocess.Popen(
            ["osascript", "-e",
             f'tell app "Terminal" to do script "{script}"'],
        )

        return json.dumps({
            "model": model_id,
            "name": model_name,
            "config_dir": str(tmp_dir),
            "status": "Launched Claude Code in new Terminal window.",
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def generate_team_config() -> str:
    """Generate an agent team config with the best free model for each role.

    Returns recommended models for Coordinator, Coder, Researcher, and Vision roles,
    each optimized for their specific capability.
    """
    try:
        raw = fetch_models()
        free = filter_free_models(raw)

        scored_models = []
        for m in free:
            if m.id == "openrouter/free":
                continue
            scored_models.append({
                "id": m.id,
                "name": m.name,
                "context": m.context_length,
                "size": format_size(parse_model_size(m.description)),
                "supports_vision": m.supports_vision,
                "scores": {
                    "agent": compute_agent_score(m),
                    "coding": compute_coding_score(m),
                    "balanced": round((compute_agent_score(m) + compute_coding_score(m) + compute_size_score(m.description)) / 3, 1),
                },
                "strengths": _strengths(m, parse_model_size(m.description)),
            })

        def best(key):
            return max(scored_models, key=lambda m: m["scores"][key])

        # Find best vision model
        vision_models = [m for m in scored_models if m["supports_vision"]]
        best_vision = max(vision_models, key=lambda m: m["context"]) if vision_models else scored_models[0]

        top = best("agent")
        team = {
            "coordinator": {
                "model": top["id"],
                "name": top["name"],
                "reason": "Highest agent/tool-calling score — orchestrates sub-agents",
                "scores": top["scores"],
            },
            "coder": {
                "model": best("coding")["id"],
                "name": best("coding")["name"],
                "reason": "Highest coding score — handles implementation tasks",
                "scores": best("coding")["scores"],
            },
            "researcher": {
                "model": best("balanced")["id"],
                "name": best("balanced")["name"],
                "reason": "Best all-rounder for research and analysis",
                "scores": best("balanced")["scores"],
            },
            "vision": {
                "model": best_vision["id"],
                "name": best_vision["name"],
                "reason": "Best free vision-capable model for image tasks",
                "scores": best_vision["scores"],
                "context_length": best_vision["context"],
            },
        }

        # If coordinator already supports vision, note it
        if top["supports_vision"]:
            team["coordinator"]["note"] = "This model also supports vision — no separate vision agent needed."

        return json.dumps({
            "team": team,
            "note": "Configure delegation.model per role in Hermes, or use delegate_task with specific models.",
        }, indent=2, ensure_ascii=False)

    except RuntimeError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_top_free_models(criterion: str = "balanced", top_n: int = 3) -> str:
    """Evaluate free models and return top N by agent/coding/size capability.

    Three scoring dimensions (each 0-100):
      - Agent: tools(+35), structured_outputs(+30), reasoning(+20), tool_choice(+10), parallel_tool_calls(+5)
      - Coding: context_length(0-30), tools(+25), reasoning(+25), structured_outputs(+20)
      - Size: parsed from description (1T+=100, 400B+=90, 100B+=75, 30B+=55, 10B+=35, 1B+=15)
      - Balanced: average of all three

    Args:
        criterion: "agent", "coding", "size", or "balanced"
        top_n: Number of results to return (1-10)
    """
    top_n = max(1, min(top_n, 10))
    try:
        raw = fetch_models()
        free = filter_free_models(raw)

        scored: list[tuple[float, ModelInfo, float, float, float]] = []
        for m in free:
            if m.id == "openrouter/free":
                continue
            agent = compute_agent_score(m)
            coding = compute_coding_score(m)
            size = compute_size_score(m.description)
            balanced = (agent + coding + size) / 3
            key = {"agent": agent, "coding": coding, "balanced": balanced, "size": size}.get(criterion, balanced)
            scored.append((key, m, agent, coding, size))

        scored.sort(key=lambda x: x[0], reverse=True)

        top = []
        needs_vision_companion = False
        for i, (_, m, agent, coding, size) in enumerate(scored[:top_n], 1):
            if not m.supports_vision:
                needs_vision_companion = True
            top.append({
                "rank": i,
                "id": m.id,
                "name": m.name,
                "context_length": m.context_length,
                "model_size": format_size(parse_model_size(m.description)),
                "supports_vision": m.supports_vision,
                "strengths": _strengths(m, parse_model_size(m.description)),
                "scores": {
                    "agent": agent,
                    "coding": coding,
                    "size": size,
                    "balanced": round((agent + coding + size) / 3, 1),
                },
            })

        output = {"criterion": criterion, "top_models": top}
        if needs_vision_companion:
            output["vision_note"] = (
                "Some recommended models do not support vision. "
                "If you need image recognition, pair with a vision-capable model."
            )
            output["companion_vision_model"] = find_best_vision_companion(raw)

        return json.dumps(output, indent=2, ensure_ascii=False)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
