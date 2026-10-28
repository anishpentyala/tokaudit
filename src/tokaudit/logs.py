"""Parsing of LLM request logs into normalized Request records.

Supported line formats (one JSON object per line, JSONL):

- OpenAI chat style: ``{"model": ..., "messages": [{"role", "content"}]}``
  where content is a string or a list of typed parts. A part carrying
  ``prompt_cache_breakpoint`` marks an explicit cache breakpoint.
- Anthropic messages style: ``{"model": "claude-...", "system": ...,
  "messages": [{"role", "content"}]}`` where content is a string or a list
  of content blocks. A block carrying ``cache_control`` marks an explicit
  cache breakpoint. A top-level ``system`` becomes the first message.
- Generic: ``{"prompt": "..."}`` (or ``text``) becomes a single user
  message.

Optional fields: ``id``, ``ts``/``timestamp``/``created`` (epoch seconds or
ISO-8601 string), ``provider`` (explicit override).

Prefix analysis renders each request with a deterministic canonical
template (``<role>`` headers + content, in order). Real providers render
with their own chat templates and may include hidden system content, so
absolute prefix lengths are approximate; *relative* reuse patterns across
requests from the same workload are preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .adapters import adapt


@dataclass
class Message:
    role: str
    text: str


@dataclass
class Request:
    rid: str
    model: str
    provider: str
    messages: List[Message]
    breakpoints: List[int] = field(default_factory=list)
    ts: Optional[float] = None
    session_id: Optional[str] = None
    cache_key: Optional[str] = None

    def canonical(self) -> str:
        return "\n".join(f"<{m.role}>\n{m.text}" for m in self.messages)


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _content_to_text(content: Any) -> str:
    """Flatten string or typed-part content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", None) and "text" in part:
                    texts.append(str(part["text"]))
        return "".join(texts)
    return ""


def _breakpoint_marked(content: Any) -> bool:
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and (
                "cache_control" in part or "prompt_cache_breakpoint" in part
            ):
                return True
    return False


def parse_record(obj: Dict[str, Any], idx: int = 0) -> Request:
    rid = str(obj.get("id") or f"req-{idx}")
    model = str(obj.get("model") or "unknown")
    ts = _parse_ts(obj.get("ts", obj.get("timestamp", obj.get("created"))))
    provider_hint = str(obj.get("provider") or "").lower()

    messages: List[Message] = []
    breakpoints: List[int] = []

    system = obj.get("system")
    if system is not None:
        messages.append(Message(role="system", text=_content_to_text(system)))
        if _breakpoint_marked(system):
            breakpoints.append(len(messages) - 1)

    raw_messages = obj.get("messages")
    if raw_messages is None:
        prompt = obj.get("prompt", obj.get("text"))
        if prompt is None:
            raise ValueError(f"record {rid}: no messages/prompt field")
        raw_messages = [{"role": "user", "content": prompt}]

    for m in raw_messages:
        if not isinstance(m, dict):
            raise ValueError(f"record {rid}: message is not an object")
        messages.append(
            Message(role=str(m.get("role", "user")), text=_content_to_text(m.get("content")))
        )
        if _breakpoint_marked(m.get("content")):
            breakpoints.append(len(messages) - 1)

    if provider_hint in ("openai", "anthropic"):
        provider = provider_hint
    elif system is not None or model.lower().startswith(("claude",)):
        provider = "anthropic"
    elif model.lower().startswith(("gpt", "o1", "o3", "o4")):
        provider = "openai"
    else:
        provider = "generic"

    return Request(
        rid=rid, model=model, provider=provider,
        messages=messages, breakpoints=breakpoints, ts=ts,
        session_id=(str(obj["session_id"]) if obj.get("session_id") else None),
        cache_key=(
            str(obj.get("prompt_cache_key") or obj.get("cache_key"))
            if (obj.get("prompt_cache_key") or obj.get("cache_key"))
            else None
        ),
    )


def load_log(path: str) -> List[Request]:
    requests: List[Request] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{idx + 1}: invalid JSON: {exc}")
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{idx + 1}: expected a JSON object per line")
            try:
                requests.append(parse_record(adapt(obj), idx))
            except ValueError as exc:
                raise SystemExit(f"{path}:{idx + 1}: {exc}")
    if not requests:
        raise SystemExit(f"no requests found in {path}")
    return requests
