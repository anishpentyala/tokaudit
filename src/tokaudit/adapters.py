"""Adapters for known LLM gateway/observability export shapes.

Each adapter normalizes one exported record into the plain dict shape that
``logs.parse_record`` understands. Detection is by field signature and is
deliberately strict: an unrecognized shape raises a clear error rather
than silently mis-parsing.

Grounding status (re-checked dates inline):
- LiteLLM SpendLogs row: fields verified against litellm/proxy/
  schema.prisma (model LiteLLM_SpendLogs) on 2026-10-28, re-verified same
  day against BerriAI/litellm main schema.prisma (request_id, startTime,
  messages, model, custom_llm_provider, session_id all present).
- Langfuse observation (Observations API v2, GET /api/public/v2/observations):
  validated 2026-10-28 against the authoritative Fern schema
  (github.com/langfuse/langfuse, fern/apis/server/definition/commons.yml,
  type ObservationV2; re-verified same day: core id/traceId/startTime/
  projectId/type, sessionId in basic, input/output in io, model in model
  group, all still current). Facts that shape this adapter:
  * core fields id/traceId/startTime/endTime/projectId/parentObservationId/type
    are always present; everything else is an optional field group;
  * `input`/`output` belong to the optional `io` group and are returned as
    raw strings (JSON-serialized when structured);
  * `type` distinguishes GENERATION (an LLM call) from SPAN/EVENT/etc.
  Loud errors on mismatch.
- Helicone request row (POST /v1/request/query): validated 2026-10-28
  against the authoritative HeliconeRequest type
  (github.com/Helicone/helicone, packages/llm-mapper/types.ts;
  re-verified same day, fields unchanged). Rows are
  FLAT: request_id, request_created_at, request_body (raw request JSON),
  response_body, provider, request_model/response_model. There is no nested
  request/response wrapper. request_body may arrive as a JSON string.
  Loud errors on mismatch.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _is_litellm_spendlog(obj: Dict[str, Any]) -> bool:
    return "request_id" in obj and "startTime" in obj and "messages" in obj


def _is_langfuse_observation(obj: Dict[str, Any]) -> bool:
    # Core ObservationV2 fields; `input` is NOT required for detection because
    # it sits in the optional `io` field group of the v2 export.
    return (
        "id" in obj
        and ("startTime" in obj or "start_time" in obj)
        and ("traceId" in obj or "projectId" in obj or "type" in obj)
    )


def _is_helicone_row(obj: Dict[str, Any]) -> bool:
    # Flat HeliconeRequest row; `request_id` alone collides with LiteLLM
    # spend logs, which are matched first, so require the body field too.
    return "request_id" in obj and "request_body" in obj


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
        obs_type = obj.get("type")
        if isinstance(obs_type, str) and obs_type != "GENERATION":
            raise ValueError(
                f"langfuse observation {obj.get('id')}: type {obs_type} is not an "
                "LLM generation; filter the export to GENERATION observations"
            )
        if "input" not in obj:
            raise ValueError(
                f"langfuse observation {obj.get('id')}: no input field; re-export "
                "with the `io` field group so the observation input is included"
            )
        inp = obj["input"]
        out = {
            "id": obj.get("id"),
            "model": obj.get("model"),
            "ts": obj.get("startTime", obj.get("start_time")),
            "session_id": obj.get("sessionId", obj.get("session_id")),
        }
        # The v2 API returns input as a raw string; structured inputs arrive
        # JSON-serialized, so try to recover messages from a string first.
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except (json.JSONDecodeError, TypeError):
                out["prompt"] = inp
                return {k: v for k, v in out.items() if v is not None}
        if isinstance(inp, list):
            out["messages"] = inp
        elif isinstance(inp, dict) and isinstance(inp.get("messages"), list):
            out["messages"] = inp["messages"]
            if inp.get("system") is not None:
                out["system"] = inp["system"]
        elif isinstance(inp, str):
            out["prompt"] = inp
        else:
            raise ValueError(
                f"langfuse observation {out['id']}: input has unexpected shape "
                f"({type(inp).__name__}); export the observation's messages"
            )
        return {k: v for k, v in out.items() if v is not None}

    if _is_helicone_row(obj):
        body = obj.get("request_body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = None
        out = {
            "id": obj.get("request_id"),
            "model": obj.get("request_model") or obj.get("model"),
            "ts": obj.get("request_created_at", obj.get("created_at")),
            "provider": obj.get("provider"),
        }
        if isinstance(body, dict) and isinstance(body.get("messages"), list):
            out["messages"] = body["messages"]
            if body.get("system") is not None:
                out["system"] = body["system"]
        elif isinstance(body, dict) and isinstance(body.get("prompt"), str):
            out["prompt"] = body["prompt"]
        else:
            raise ValueError(
                f"helicone row {out['id']}: request_body has no messages/prompt "
                f"(got {type(obj.get('request_body')).__name__})"
            )
        # Provider-reported usage (typed fields on HeliconeRequest, verified
        # 2026-10-28): carried through for the report's simulated-vs-reported
        # comparison; never fed back into the simulation.
        usage = {}
        for src_key, dst_key in (
            ("prompt_tokens", "prompt_tokens"),
            ("prompt_cache_read_tokens", "cache_read_tokens"),
            ("prompt_cache_write_tokens", "cache_write_tokens"),
        ):
            v = obj.get(src_key)
            if isinstance(v, (int, float)) and v is not None:
                usage[dst_key] = int(v)
        if usage:
            out["actual_usage"] = usage
        return {k: v for k, v in out.items() if v is not None}

    return obj
