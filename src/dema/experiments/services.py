"""Lifecycle management for model servers used by experiment methods.

The experiment runner remains a pure client.  This module is the small bridge
used by the top-level CLI: it reuses an already healthy endpoint, otherwise it
starts the matching server script, waits until a real inference succeeds, and
stops only the process that it started.
"""

from __future__ import annotations

import copy
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from ..utils.config import REPO_ROOT, Config
from ..utils.logging import get_console_logger
from .preflight import check_decision, check_qwen

log = get_console_logger("dema.services")

METHOD_SERVICE = {
    "magneto_qwen": "qwen",
    "dema": "decision",
    "dema_no_rerank": "decision",
    "dema_no_struct": "decision",
    "dema_decision": "decision",
    "dema_fusion": "decision",
    "dema_jev_weight": "decision",
    "dema_jina_rerank": "decision",
    "dema_jina_no_coma": "decision",
    "dema_shared": "decision",
    "dema_single": "decision",
    "dema_own_retrieval": "decision",
    "dema_legacy": "decision",
}


def required_service(method: str) -> str | None:
    """Return the external service required by *method*, if any."""
    return METHOD_SERVICE.get(method)


class ServiceError(RuntimeError):
    pass


class ManagedService:
    """Context manager for one Qwen or decision endpoint."""

    def __init__(self, name: str, config: Config, gpu: str = "0", timeout: float = 900):
        if name not in ("qwen", "decision"):
            raise ValueError(f"unknown service {name!r}")
        self.name = name
        self.config = config
        self.gpu = str(gpu)
        self.timeout = float(timeout)
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    @property
    def url(self) -> str:
        section = "qwen" if self.name == "qwen" else "decision"
        return str(self.config.models[section]["base_url"])

    def _port_is_open(self) -> bool:
        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    def _health_errors(self) -> list[str]:
        # Readiness requests should fail quickly while a server is still loading;
        # the main experiment retains its longer inference timeout.
        probe = copy.deepcopy(self.config)
        probe.models[self.name]["timeout"] = 15
        probe.models[self.name]["max_retries"] = 0
        return check_qwen(probe) if self.name == "qwen" else check_decision(probe)

    def start(self) -> "ManagedService":
        if self._port_is_open():
            errors = self._health_errors()
            if errors:
                raise ServiceError(
                    f"port for {self.name} is already occupied, but the endpoint is unhealthy: {errors[0]}"
                )
            log.info("using existing %s service at %s", self.name, self.url)
            return self

        script = REPO_ROOT / "scripts" / f"serve_{self.name}.sh"
        if not script.is_file():
            raise ServiceError(f"server launcher not found: {script}")
        log_path = self.config.logs_dir / "servers" / f"{self.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        if self.name == "qwen":
            vllm_python = Path.home() / ".venvs" / "vllm" / "bin" / "python"
            if not vllm_python.is_file():
                raise ServiceError(
                    f"vLLM interpreter not found at {vllm_python}; run scripts/setup_vllm.sh"
                )
            env.setdefault("QWEN_PYTHON", str(vllm_python))
            env.setdefault("QWEN_BACKEND", "vllm")
        log.info(
            "starting %s service on GPU %s; server log: %s", self.name, self.gpu, log_path
        )
        self.process = subprocess.Popen(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self

    def wait_ready(self) -> None:
        if self.process is None:  # existing endpoint was already checked
            return
        deadline = time.monotonic() + self.timeout
        last_error = "endpoint has not opened its port"
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise ServiceError(
                    f"{self.name} server exited with code {code}; see "
                    f"{self.config.logs_dir / 'servers' / (self.name + '.log')}"
                )
            if self._port_is_open():
                errors = self._health_errors()
                if not errors:
                    log.info("%s service is ready at %s", self.name, self.url)
                    return
                last_error = errors[0]
            time.sleep(5)
        raise ServiceError(
            f"{self.name} service was not ready after {self.timeout:g}s: {last_error}; "
            f"see {self.config.logs_dir / 'servers' / (self.name + '.log')}"
        )

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            log.info("stopping %s service (pid %d)", self.name, process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "ManagedService":
        try:
            self.start()
            self.wait_ready()
            return self
        except Exception:
            self.stop()
            raise

    def __exit__(self, *_exc) -> None:
        self.stop()
