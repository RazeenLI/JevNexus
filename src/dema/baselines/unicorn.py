"""Unicorn (Tu et al., SIGMOD 2023) zero-shot schema matching inference.

Re-implements only what inference needs: the DeBERTa-base encoder ([CLS]
feature), the mixture-of-experts layer and the binary classifier, loaded from
the released pre-trained ``UnicornPlus`` checkpoint
(``RUC-DataLab/unicorn-plus-v1`` on Hugging Face, pinned revision). Module names
mirror the original so the released state dicts load with ``strict=True``.

Input serialization follows the reference used in Magneto's comparison: every
column becomes ``[ATT] <name> [VAL] v1 [VAL] v2 ...`` with the first 20 unique
values (in order of appearance), pairs are encoded as
``[CLS] source [SEP] target [SEP]`` with the reference truncation rule and
``max_seq_length=128``. Every (source, target) pair is scored; the score is the
match logit (reference ``predict_moe``), or optionally the softmax probability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.device import resolve_device
from .base import ScoreMatrixBaseline


def serialize_column(name: str, series: pd.Series, max_values: int) -> str:
    values = list(series.unique()[:max_values])
    return f"[ATT] {name} [VAL] {' [VAL] '.join(str(x) for x in values)}"


def encode_pair(left: str, right: str, tokenizer, max_len: int, cls_token="[CLS]", sep_token="[SEP]"):
    """Reference ``convert_one_example_to_features_sep`` truncation."""
    ltokens = tokenizer.tokenize(left)
    rtokens = tokenizer.tokenize(right)
    more = len(ltokens) + len(rtokens) - max_len + 3
    if more > 0:
        if more < len(rtokens):
            rtokens = rtokens[: len(rtokens) - more]
        elif more < len(ltokens):
            ltokens = ltokens[: len(ltokens) - more]
        else:
            rtokens, ltokens = rtokens[:50], ltokens[:50]
    tokens = [cls_token] + ltokens + [sep_token] + rtokens + [sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    return ids + [0] * pad, mask + [0] * pad


def build_modules(encoder_name: str, size_output: int, units: int, experts: int):
    import torch
    from torch import nn
    from transformers import DebertaModel

    class DebertaBaseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DebertaModel.from_pretrained(encoder_name)

        def forward(self, x, mask):
            return self.encoder(x, attention_mask=mask).last_hidden_state[:, 0, :]

    class MoEModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.05)
            self.expert_kernels = nn.ModuleList(
                [nn.Sequential(nn.Linear(size_output, units), nn.BatchNorm1d(units), nn.LeakyReLU())
                 for _ in range(experts)]
            )
            self.gate_kernel = nn.Sequential(
                nn.Linear(size_output, units), nn.LeakyReLU(), nn.Linear(units, experts), nn.LeakyReLU()
            )

        def forward(self, x):
            expert_out = torch.stack([k(x) for k in self.expert_kernels], 0).permute(1, 2, 0)
            gate = torch.softmax(self.gate_kernel(x), dim=-1).unsqueeze(1)
            return torch.sum(expert_out * gate.expand_as(expert_out), 2)

    class MOEClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(units, 2)

        def forward(self, x):
            return self.classifier(self.dropout(x))

    return DebertaBaseEncoder(), MoEModule(), MOEClassifier()


class UnicornMatcher(ScoreMatrixBaseline):
    name = "unicorn"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._modules = None
        self._tokenizer = None
        self._device = None

    def describe(self) -> dict[str, Any]:
        return {"method": self.name, "checkpoint_repo": self.cfg.get("checkpoint_repo"),
                "checkpoint_revision": self.cfg.get("checkpoint_revision"), "config": self.cfg}

    def _checkpoint_files(self) -> dict[str, Path]:
        prefix = self.cfg.get("checkpoint_prefix", "UnicornPlus")
        names = {part: f"{prefix}_{part}.pt" for part in ("encoder", "moe", "cls")}
        local = self.cfg.get("checkpoint_dir")
        if local:
            paths = {k: Path(local) / v for k, v in names.items()}
            missing = [str(p) for p in paths.values() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Unicorn checkpoint files missing: {missing}")
            return paths
        from huggingface_hub import hf_hub_download

        return {
            k: Path(hf_hub_download(self.cfg["checkpoint_repo"], v, revision=self.cfg.get("checkpoint_revision")))
            for k, v in names.items()
        }

    def load(self) -> None:
        if self._modules is not None:
            return
        import torch
        from transformers import DebertaTokenizer

        cfg = self.cfg
        self._device = resolve_device(cfg.get("device", "auto"))
        self._tokenizer = DebertaTokenizer.from_pretrained(cfg["encoder"])
        encoder, moe, cls = build_modules(
            cfg["encoder"], int(cfg.get("size_output", 768)), int(cfg.get("units", 768)),
            int(cfg.get("expertsnum", 6)),
        )
        files = self._checkpoint_files()
        for module, key in ((encoder, "encoder"), (moe, "moe"), (cls, "cls")):
            state = torch.load(files[key], map_location="cpu", weights_only=True)
            # Older transformers versions stored this constant buffer; it is not a weight.
            state = {k: v for k, v in state.items() if not k.endswith("embeddings.position_ids")}
            module.load_state_dict(state, strict=True)
            module.to(self._device).eval()
        self._modules = (encoder, moe, cls)

    def compute_scores(self, source_df, target_df):
        import torch

        self.load()
        encoder, moe, cls = self._modules
        max_values = int(self.cfg.get("max_values", 20))
        max_len = int(self.cfg.get("max_seq_length", 128))
        batch_size = int(self.cfg.get("batch_size", 128))
        src_cols = [str(c) for c in source_df.columns]
        tgt_cols = [str(c) for c in target_df.columns]
        src_text = [serialize_column(c, source_df.iloc[:, i], max_values) for i, c in enumerate(src_cols)]
        tgt_text = [serialize_column(c, target_df.iloc[:, j], max_values) for j, c in enumerate(tgt_cols)]
        # Reference joins with " [SEP] " and splits on "[SEP]": left keeps a
        # trailing space, right a leading one.
        pairs = [(s, t) for s in range(len(src_cols)) for t in range(len(tgt_cols))]
        use_prob = self.cfg.get("score", "match_logit") == "match_probability"
        scores: dict[tuple[str, str], float] = {}
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start : start + batch_size]
                encoded = [encode_pair(src_text[s] + " ", " " + tgt_text[t], self._tokenizer, max_len)
                           for s, t in chunk]
                ids = torch.tensor([e[0] for e in encoded], device=self._device)
                mask = torch.tensor([e[1] for e in encoded], device=self._device)
                logits = cls(moe(encoder(ids, mask)))
                values = torch.softmax(logits, dim=-1)[:, 1] if use_prob else logits[:, 1]
                for (s, t), v in zip(chunk, values.float().cpu().tolist()):
                    scores[(src_cols[s], tgt_cols[t])] = float(v)
        return scores
