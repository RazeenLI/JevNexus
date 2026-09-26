# Vendored: original Magneto

Source: https://github.com/VIDA-NYU/magneto-matcher, commit
`6620623265fc7feac0f053996e62b68a13a72a57`, directory `algorithms/magneto/magneto`
(Apache-2.0, see LICENSE). Files are byte-identical to that commit except:

1. `magneto/llm_reranker.py` line 3: `from litellm import completion` ->
   `from magneto._local_llm import completion` (LLM backend swap only; prompt,
   parsing with json_repair, 5 attempts and fallback behaviour are unchanged).
2. `magneto/_local_llm.py` (new): forwards Magneto's chat messages and
   `llm_model_kwargs` to the local OpenAI-compatible Qwen3.5-9B endpoint.

Verify with: `diff -r` against the upstream commit.
