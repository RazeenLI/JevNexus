from __future__ import annotations

import copy
import json
import re
import sys
import zlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dema.utils.config import load_config  # noqa: E402


def trigram_encoder(texts):
    """Deterministic bag-of-character-trigrams embedding (no model download)."""
    dim = 512
    out = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        name = text.splitlines()[0].lower()
        for j in range(len(name) - 2):
            out[i, zlib.crc32(name[j:j + 3].encode()) % dim] += 1.0
        out[i, 0] += 1e-3  # avoid all-zero rows
    return out


def write_mini_raw(raw: Path) -> None:
    """Tiny raw benchmark in the GDC-SM and Valentine layouts."""
    gdc = raw / "gdc" / "data"
    for sub in ("ground-truth", "source-tables", "target-tables"):
        (gdc / sub).mkdir(parents=True, exist_ok=True)
    (gdc / "source-tables" / "Toy.csv").write_text(
        "Patient Age,Sex,Tumor Grade\n34,M,G1\n51,F,G2\n,F,G3\n", encoding="utf-8")
    (gdc / "target-tables" / "gdc_unique_columns_concat_values.csv").write_text(
        "age_at_diagnosis,gender,tumor_grade,country_of_birth,days_to_birth\n"
        "34,male,G1,France,-12000\n51,female,G2,Spain,-18000\n", encoding="utf-8")
    (gdc / "ground-truth" / "Toy.csv").write_text(
        "original_paper_variable_names,GDC_format_variable_names\n"
        "Patient Age,age_at_diagnosis\nPatient Age,days_to_birth\nSex,gender\nTumor Grade,tumor_grade\n",
        encoding="utf-8")
    case = raw / "valentine" / "Valentine-datasets" / "OpenData" / "Joinable" / "toy_ac1"
    case.mkdir(parents=True, exist_ok=True)
    # latin-1 encoded source to exercise encoding normalization
    (case / "toy_ac1_source.csv").write_bytes("city,café_name,rating\r\nParis,Chez A,4.5\r\nOslo,B,3\r\n".encode("latin-1"))
    (case / "toy_ac1_target.csv").write_text("town,restaurant,stars,zip\nRome,C,5,00100\nLyon,D,2,69001\n",
                                             encoding="utf-8")
    (case / "toy_ac1_mapping.json").write_text(json.dumps({"matches": [
        {"source_table": "toy_ac1_source", "source_column": "city", "target_table": "toy_ac1_target",
         "target_column": "town"},
        {"source_table": "toy_ac1_source", "source_column": "café_name", "target_table": "toy_ac1_target",
         "target_column": "restaurant"},
    ]}), encoding="utf-8")


@pytest.fixture
def mini_benchmark(config, tmp_path):
    from dema.data.prepare import run as prepare_run

    raw = tmp_path / "raw"
    write_mini_raw(raw)
    config.paths["raw_data"] = str(raw)
    config.paths["processed_data"] = str(tmp_path / "processed")
    config.paths["manifest"] = str(tmp_path / "manifest.jsonl")
    config.experiment["datasets"] = ["GDC", "OpenData"]
    records = prepare_run(config, ["GDC", "OpenData"])
    return config, records


@pytest.fixture
def config(tmp_path):
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg.paths["saves"] = str(tmp_path / "saves")
    cfg.paths["logs"] = str(tmp_path / "logs")
    cfg.paths["metrics"] = str(tmp_path / "metrics")
    cfg.paths["cache"] = str(tmp_path / "cache")
    return cfg


def _name_sim(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    ta = {a[i:i + 3] for i in range(max(1, len(a) - 2))}
    tb = {b[i:i + 3] for i in range(max(1, len(b) - 2))}
    return len(ta & tb) / max(1, len(ta | tb))


class FakeServer:
    """Threaded HTTP server; ``mode`` controls the behaviour of the fake model."""

    def __init__(self, kind: str):
        self.kind = kind
        self.mode = "ok"
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, {"status": "ok", "data": []})

            def _send(self, code, body, raw=None):
                data = raw.encode() if raw is not None else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(req)
                if server.kind == "qwen":
                    self._qwen(req)
                else:
                    self._decision(req)

            def _qwen(self, req):
                user = req["messages"][1]["content"]
                source_match = re.search(r"Candidate Column: Column: ([^,\n]+)", user)
                src = source_match.group(1) if source_match else "a"
                names = re.findall(r"^Column: ([^,\n]+)", user, re.MULTILINE)
                results = [{"column": name, "score": round(_name_sim(src, name), 4)} for name in names]
                if server.mode == "invalid_json":
                    content = "not json"
                elif server.mode == "missing":
                    content = json.dumps(results[:-1])
                else:
                    content = json.dumps(results or [{"column": "b", "score": 1.0}])
                self._send(200, {"choices": [{"message": {"content": content}}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

            def _decision(self, req):
                state = req["state"]
                src = re.search(r"Source column:\nname: (.*)", state).group(1)
                names = dict(re.findall(r"\[(c\d+)\]\nname: (.*)", state))
                if not names:
                    names = {
                        qid: re.search(r"name: ([^\n]+)", question["instructions"]).group(1)
                        for qid, question in req["questions"].items()
                    }
                answers = {q: {"type": "noul", "noul": round(_name_sim(src, names[q]), 4)} for q in req["questions"]}
                if server.mode == "out_of_range":
                    first = next(iter(answers))
                    answers[first]["noul"] = 1.5
                self._send(200, {"answers": answers, "usage": {"input_tokens": 7, "output_tokens": 0}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_servers(config):
    qwen, decision = FakeServer("qwen"), FakeServer("decision")
    config.models["qwen"]["base_url"] = f"http://127.0.0.1:{qwen.port}/v1"
    config.models["qwen"]["max_retries"] = 1
    config.models["decision"]["base_url"] = f"http://127.0.0.1:{decision.port}"
    config.models["decision"]["max_retries"] = 1
    yield qwen, decision
    qwen.close()
    decision.close()
