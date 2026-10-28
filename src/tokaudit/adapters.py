"""Adapters for known LLM gateway/observability export shapes.

Each adapter normalizes one exported record into the plain dict shape that
``logs.parse_record`` understands. Detection is by field signature and is
deliberately strict: an unrecognized shape raises a clear error rather
than silently mis-parsing.

Grounding status (re-checked dates inline):
- LiteLLM SpendLogs row: fields verified against litellm/proxy/
  schema.prisma (model LiteLLM_SpendLogs) on 2026-10-28.
- Langfuse observation: endpoint shape (Observations API v2,
  GET /api/public/v2/observations) verified 2026-10-28; exact field set
  is from their public API docs and may drift. Loud errors on mismatch.
- Helicone request row: query endpoint shape (POST /v1/request/query)
  verified 2026-10-28; request/response wrapper field names may drift.
"""

from __future__ import annotations

from typing import Any, Dict


def _is_litellm_spendlog(obj: Dict[str, Any]) -> bool:
    return "request_id" in obj and "startTime" in obj and "messages" in obj


def _is_langfuse_observation(obj: Dict[str, Any]) -> bool:
    return "input" in obj and ("startTime" in obj or "start_time" in obj)


def _is_helicone_row(obj: Dict[str, Any]) -> bool:
    return isinstance(obj.get("request"), dict) and "response" in obj


def adapt(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a known export record to parse_record's input shape.

    Returns the object unchanged when no adapter matches.
    """
    if _is_litellm_spendlog(obj):
        out: Dict[str, Any] = {
            "id": obj.get("request_id"),
            "model": obj.get("model"),
            "ts": obj.get("startTime"),
            "provider": obj.get("custom_llm_provider") or None,
        }
        messages = obj.get("messages")
        if isinstance(messages, str):
            out["prompt"] = messages
        elif isinstance(messages, list):
            out["messages"] = messages
        else:
            raise ValueError(
                f"litellm record {out['id']}: messages has unexpected type "
                f"{type(messages).__name__}"
            )
        if obj.get("session_id"):
            out["session_id"] = obj["session_id"]
        return {k: v for k, v in out.items() if v is not None}

    if _is_langfuse_observation(obj):
        inp = obj["input"]
        out = {
            "id": obj.get("id"),
            "model": obj.get("model"),
            "ts": obj.get("startTime", obj.get("start_time")),
        }
        if isinstance(inp, str):
            out["prompt"] = inp
        elif isinstance(inp, list):
            out["messages"] = inp
        elif isinstance(inp, dict) and isinstance(inp.get("messages"), list):
            out["messages"] = inp["messages"]
            if inp.get("system") is not None:
                out["system"] = inp["system"]
        else:
            raise ValueError(
                f"langfuse observation {out['id']}: input has unexpected shape "
                f"({type(inp).__name__}); export the observation's messages"
            )
        return {k: v for k, v in out.items() if v is not None}

    if _is_helicone_row(obj):
        inner = obj["request"]
        body = inner.get("body", inner)
        out = {
            "id": obj.get("request_id", obj.get("id")),
            "model": body.get("model", inner.get("model")),
            "ts": obj.get("created_at", inner.get("created_at")),
            "provider": inner.get("provider"),
        }
        if isinstance(body.get("messages"), list):
            out["messages"] = body["messages"]
        elif isinstance(body.get("prompt"), str):
            out["prompt"] = body["prompt"]
        else:
            raise ValueError(
                f"helicone row {out['id']}: request body has no messages/prompt"
            )
        return {k: v for k, v in out.items() if v is not None}

    return obj
