"""ISResMat — In Situ Neural Relational Schema Matcher (Du et al., ICDE 2024).

Independent re-implementation of the matching inference behaviour of the
reference code used in Magneto's comparison (``process_mode=0``: train-to-match
on the case itself). Per case:

1. impute missing cells by sampling observed values of the same column, drop
   all-empty columns, shuffle rows (all seeded);
2. repeatedly sample *pairwise fragments* of each table: ``frag_width`` columns,
   two disjoint-in-time row samples of ``frag_height`` rows; every column of a
   fragment becomes ``[CLS] v1 v2 ... [SEP]`` where each value is replaced by the
   column name with probability ``col_name_prob``; numerical columns get the
   bucket random-walk "fingerprint" appended;
3. fine-tune BERT + projector with (a) a contrastive loss between the two
   fragments of the same column, (b) a student-t self-assignment loss to one
   learnable *agent* per column, and (c) the matching rectification loss between
   the agent similarity matrix and its Sinkhorn optimal-transport version;
4. rank targets by the final agent similarity matrix ``src_agents @ tgt_agents.T``.

Simplifications (documented in README): schema-name transformations
(``schema_process_type`` / ``col_name_variant_prob``) are not implemented
(the reference default ``00`` uses original names); no validation/early
stopping; fragments longer than 512 tokens are truncated instead of resampled;
the complete ranking is kept instead of the reference top-10 truncation.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

import numpy as np
import pandas as pd

from ..utils.device import resolve_device
from .base import ScoreMatrixBaseline


class ISResMatMatcher(ScoreMatrixBaseline):
    name = "isresmat"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._tokenizer = None
        self._bert_state = None
        self._bert_config = None

    def load(self) -> None:
        if self._tokenizer is not None:
            return
        from transformers import BertModel, BertTokenizer

        name = self.cfg.get("encoder", "bert-base-uncased")
        self._tokenizer = BertTokenizer.from_pretrained(name)
        bert = BertModel.from_pretrained(name)
        # Keep pristine pretrained weights in memory: every case starts from them.
        self._bert_config = bert.config
        self._bert_state = {k: v.detach().clone() for k, v in bert.state_dict().items()}

    def describe(self) -> dict[str, Any]:
        return {"method": self.name, "encoder": self.cfg.get("encoder"), "config": self.cfg}

    # --------------------------------------------------------- preprocessing
    @staticmethod
    def _prepare_table(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        empty = [c for c in df.columns if (df[c].isna() | (df[c].astype(str) == "")).all()]
        df = df.drop(columns=empty)
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                series = series.replace("", np.nan)
            observed = series.dropna()
            pool = observed.to_numpy() if pd.api.types.is_numeric_dtype(series) else observed.unique()
            mask = series.isna().to_numpy()
            if mask.any() and len(pool):
                filled = series.to_numpy(dtype=object).copy()
                filled[mask] = rng.choice(pool, size=int(mask.sum()))
                series = pd.Series(filled, index=series.index)
                if pd.api.types.is_numeric_dtype(observed):
                    series = pd.to_numeric(series)
            df[col] = series
        order = rng.permutation(len(df))
        return df.iloc[order].reset_index(drop=True)

    # ------------------------------------------------------------- fragments
    class _Table:
        def __init__(self, df: pd.DataFrame, cfg: dict, rng: random.Random, np_rng: np.random.Generator):
            self.df = df
            self.cols = list(df.columns)
            self.label = {c: i for i, c in enumerate(self.cols)}
            self.cfg = cfg
            self.rng = rng
            self.np_rng = np_rng
            self.bins: dict[str, np.ndarray] = {}
            self.bin_probs: dict[str, np.ndarray] = {}
            n_bins = int(cfg.get("numerical_col_bins", 20))
            if n_bins > 0:
                for c in self.cols:
                    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                        codes = pd.cut(df[c], bins=n_bins, labels=False)
                        counts = np.bincount(codes.dropna().astype(int), minlength=n_bins).astype(float)
                        self.bins[c] = codes.to_numpy()
                        self.bin_probs[c] = counts / counts.sum() if counts.sum() else counts

        def _fingerprint(self, col: str, row: int) -> str:
            probs = self.bin_probs[col]
            start = self.bins[col][row]
            if start is None or (isinstance(start, float) and math.isnan(start)):
                return ""
            walk = [int(start)]
            for _ in range(int(self.cfg.get("numerical_col_window_size", 10)) - 1):
                cur = walk[-1]
                vec = np.zeros(len(probs))
                if cur > 0:
                    vec[cur - 1] = probs[cur - 1]
                if cur < len(probs) - 1:
                    vec[cur + 1] = probs[cur + 1]
                if vec.sum() == 0:
                    break
                walk.append(int(self.np_rng.choice(len(probs), p=vec / vec.sum())))
            return " ".join(str(i) for i in walk)

        def column_sentence(self, col: str, rows: list[int]) -> str:
            values = []
            name_prob = float(self.cfg.get("col_name_prob", 0.5))
            for r in rows:
                text = str(self.df.at[r, col])
                if col in self.bins:
                    # Reference behaviour: the fingerprint is concatenated directly.
                    text = text + self._fingerprint(col, r)
                if name_prob > 0 and self.rng.random() < name_prob:
                    text = col
                values.append(text)
            return "[CLS] " + " ".join(values) + " [SEP]"

        def sample(self, frag_width: int, frag_height: int) -> tuple[list[str], list[int]]:
            n_rows = len(self.df)
            h = min(frag_height, n_rows)
            first = self.rng.sample(range(n_rows), h)
            second = self.rng.sample(range(n_rows), h)
            cols = self.rng.sample(self.cols, frag_width)
            sentences, labels = [], []
            for rows in (first, second):
                order = cols[:]
                self.rng.shuffle(order)
                for c in order:
                    sentences.append(self.column_sentence(c, rows))
                    labels.append(self.label[c])
            return sentences, labels

    # ----------------------------------------------------------------- model
    def _build_model(self, n_src: int, n_tgt: int, device: str):
        import torch
        import torch.nn.functional as F
        from torch import nn
        from transformers import BertModel

        cfg = self.cfg
        bert = BertModel(copy.deepcopy(self._bert_config))
        bert.load_state_dict(self._bert_state)
        dim = int(cfg.get("output_repr_dim", 512))
        std = float(cfg.get("cols_repr_init_std", 0.0005))
        scale = float(cfg.get("t_dist_scale_factor", 10))

        class Agents(nn.Module):
            def __init__(self, n):
                super().__init__()
                init = torch.zeros(n, dim)
                nn.init.normal_(init, mean=0.0, std=std)
                self._agents = nn.Parameter(init)

            @property
            def agents(self):
                return F.normalize(self._agents, dim=1)

            def forward(self, x):
                sim = x @ self.agents.T
                ele = (scale * (1 - sim)) ** 2
                prob = 1.0 / (1 + ele)
                return prob / prob.sum(1, keepdim=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = bert
                p = bert.config.classifier_dropout
                self.dropout = nn.Dropout(p if p is not None else bert.config.hidden_dropout_prob)
                self.projector = nn.Sequential(
                    nn.Linear(bert.config.hidden_size, dim), nn.ReLU(), nn.Linear(dim, dim)
                )
                self.src_agents = Agents(n_src)
                self.tgt_agents = Agents(n_tgt)

            def embed(self, input_ids, attention_mask, cls_id):
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                idx = (input_ids == cls_id).nonzero(as_tuple=False)
                feats = out[idx[:, 0], idx[:, 1]]
                return F.normalize(self.projector(self.dropout(feats)), dim=1)

        return Model().to(device)

    @staticmethod
    def _sinkhorn(sim, lam: float, n_iter: int):
        import torch

        sm = torch.softmax(sim.reshape(-1), dim=0).reshape(sim.shape)
        r, c = sm.sum(1), sm.sum(0)
        p = torch.exp(lam * sim)
        p = p / p.sum()
        for _ in range(n_iter):
            p = p * (r / p.sum(1)).reshape(-1, 1)
            p = p * (c / p.sum(0)).reshape(1, -1)
        return p

    # ------------------------------------------------------------- matching
    def compute_scores(self, source_df, target_df):
        import torch
        import torch.nn.functional as F

        self.load()
        cfg = self.cfg
        seed = int(cfg.get("seed", 42))
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = resolve_device(cfg.get("device", "auto"))

        src = self._Table(self._prepare_table(source_df, np_rng), cfg, rng, np_rng)
        tgt = self._Table(self._prepare_table(target_df, np_rng), cfg, rng, np_rng)
        if not src.cols or not tgt.cols or len(src.df) == 0 or len(tgt.df) == 0:
            return {}
        frag_width = min(int(cfg.get("frag_width", 12)), len(src.cols), len(tgt.cols))
        frag_height = int(cfg.get("frag_height", 6))
        steps = math.ceil(
            int(cfg.get("n_trn_cols", 200)) * max(len(src.cols), len(tgt.cols)) / (frag_width * 2)
        )

        model = self._build_model(len(src.cols), len(tgt.cols), device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 3e-5)))
        tok = self._tokenizer
        cls_id = tok.cls_token_id
        max_len = int(cfg.get("max_input_length", 512))
        temperature = float(cfg.get("temperature", 1.0))
        w_meta = float(cfg.get("meta_match_loss_weight", 1.0))
        w_agent = float(cfg.get("agent_delegate_loss_weight", 1.0))
        w_rec = float(cfg.get("rec_loss_weight", 0.2))
        warmup = float(cfg.get("rec_warmup_fraction", 0.3)) * steps
        lam = float(cfg.get("sk_reg_weight", 20))
        n_iter = int(cfg.get("sk_n_iter", 5))
        kl = torch.nn.KLDivLoss(reduction="batchmean")

        def table_losses(table, agents):
            sentences, labels = table.sample(frag_width, frag_height)
            enc = tok(sentences, padding=True, truncation=True, max_length=max_len,
                      add_special_tokens=False, return_tensors="pt").to(device)
            emb = model.embed(enc["input_ids"], enc["attention_mask"], cls_id)
            lab = torch.tensor(labels, device=device)
            sim = emb @ emb.T
            same = (lab[:, None] == lab[None, :]).float()
            off = ~torch.eye(len(labels), dtype=torch.bool, device=device)
            sim = sim[off].view(len(labels), -1)
            same = same[off].view(len(labels), -1)
            contrastive = F.cross_entropy(sim / temperature, same)
            target = F.one_hot(lab, num_classes=len(table.cols)).float()
            assign = kl(agents(emb).log(), target)
            return contrastive, assign

        for step in range(1, steps + 1):
            s_con, s_asg = table_losses(src, model.src_agents)
            t_con, t_asg = table_losses(tgt, model.tgt_agents)
            sim = model.src_agents.agents @ model.tgt_agents.agents.T
            ot = self._sinkhorn(sim.detach(), lam, n_iter)
            rec = F.cross_entropy(sim.reshape(-1), ot.reshape(-1))
            rec_w = (step / warmup) * w_rec if warmup > 0 and step <= warmup else w_rec
            loss = (s_con + t_con) * w_meta + (s_asg + t_asg) * w_agent + rec * rec_w
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final = (model.src_agents.agents @ model.tgt_agents.agents.T).cpu().numpy()
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            (s, t): float(final[i, j]) for i, s in enumerate(src.cols) for j, t in enumerate(tgt.cols)
        }
