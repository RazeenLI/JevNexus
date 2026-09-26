"""DeMa patch (the only functional change to Magneto): LLM backend swap.

Replaces ``litellm.completion`` with a call to a local OpenAI-compatible
endpoint serving Qwen3.5-9B. The ``model`` argument passed by Magneto
("gpt-4o-mini") is ignored; the configured served model is used instead.
Everything Magneto sends (messages and ``llm_model_kwargs``) is forwarded
unchanged. HTTP/connection errors propagate, as they would with litellm.
"""

from types import SimpleNamespace

import requests

_CONFIG = {"base_url": None, "model": None, "timeout": 300, "extra_body": {}, "api_key": None}
STATS = {"requests": 0, "input_tokens": 0, "output_tokens": 0}


def configure(base_url, model, timeout=300, extra_body=None, api_key=None):
    _CONFIG.update(base_url=base_url.rstrip("/"), model=model, timeout=timeout,
                   extra_body=dict(extra_body or {}), api_key=api_key)


def reset_stats():
    for k in STATS:
        STATS[k] = 0


def completion(model=None, messages=None, **kwargs):
    if not _CONFIG["base_url"]:
        raise RuntimeError("local LLM endpoint not configured (call magneto._local_llm.configure)")
    payload = {"model": _CONFIG["model"], "messages": messages, **kwargs, **_CONFIG["extra_body"]}
    headers = {"Content-Type": "application/json"}
    if _CONFIG["api_key"]:
        headers["Authorization"] = f"Bearer {_CONFIG['api_key']}"
    STATS["requests"] += 1
    resp = requests.post(_CONFIG["base_url"] + "/chat/completions", json=payload,
                         headers=headers, timeout=_CONFIG["timeout"])
    resp.raise_for_status()
    body = resp.json()
    usage = body.get("usage") or {}
    STATS["input_tokens"] += int(usage.get("prompt_tokens") or 0)
    STATS["output_tokens"] += int(usage.get("completion_tokens") or 0)
    content = body["choices"][0]["message"]["content"]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
