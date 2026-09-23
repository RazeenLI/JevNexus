"""Minimal OpenAI-compatible chat-completions server for a local HF causal LM.

Used by ``scripts/serve_qwen.sh`` when vLLM is not available. It mirrors the
CoRE project's Qwen3.5-9B setup (``HFLLMClient``): bf16 weights,
``device_map="auto"``, chat template with ``enable_thinking`` (default False) and
deterministic greedy decoding. Requests are served one at a time.

    python -m dema.serving.openai_server --model Qwen/Qwen3.5-9B --port 8000

Endpoints: ``GET /health``, ``GET /v1/models``, ``POST /v1/chat/completions``.
This module is never imported by the experiment code.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Generator:
    def __init__(self, model: str, revision: str | None, dtype: str, device_map: str, max_model_len: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, dtype=torch_dtype, device_map=device_map, trust_remote_code=True
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lock = threading.Lock()

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float,
                 enable_thinking: bool) -> tuple[str, int, int]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        n_in = int(inputs["input_ids"].shape[1])
        if n_in + max_tokens > self.max_model_len:
            raise ValueError(f"prompt ({n_in} tokens) + max_tokens exceeds max_model_len {self.max_model_len}")
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        kwargs: dict[str, Any] = {"max_new_tokens": max_tokens, "pad_token_id": self.tokenizer.pad_token_id}
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[0, n_in:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, n_in, int(new_tokens.shape[0])


def make_handler(gen: Generator, served_name: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet access log; never logs bodies
            pass

        def _send(self, code: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": served_name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length))
                kwargs = req.get("chat_template_kwargs") or {}
                text, n_in, n_out = gen.generate(
                    req["messages"], int(req.get("max_tokens") or 1024), float(req.get("temperature") or 0.0),
                    bool(kwargs.get("enable_thinking", False)),
                )
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
                return
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": served_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out},
            })

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device-map", default="auto")
    p.add_argument("--max-model-len", type=int, default=32768)
    args = p.parse_args(argv)
    gen = Generator(args.model, args.revision, args.dtype, args.device_map, args.max_model_len)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(gen, args.served_model_name or args.model))
    print(f"serving {args.model} on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
