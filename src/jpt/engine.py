"""Start inference servers and talk to them over HTTP.

Engines:
* ``llama.cpp`` — ``llama-server`` with a GGUF file. Native ``/completion``
  endpoint, which returns the server's own timings (prefill / decode).
* ``mlx`` — ``mlx_lm.server`` with an MLX repo. OpenAI-compatible endpoints;
  timings are measured on the client side.

Both expose ``/v1/chat/completions``, used for the quality benchmark.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .memory import wired_bytes


# --------------------------------------------------------------------------- results


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    gen_tokens: int
    ttft_s: float  # request sent -> first token received
    wall_s: float  # request sent -> last token received
    prefill_tps: float | None
    decode_tps: float | None
    timing_source: str  # "server" or "client"
    token_times: list[float] = field(default_factory=list, repr=False)


# --------------------------------------------------------------------------- HTTP helpers


def _post(url: str, payload: dict, timeout: float = 600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _post_json(url: str, payload: dict, timeout: float = 600) -> dict:
    with _post(url, payload, timeout) as r:
        return json.loads(r.read())


def _get_json(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def iter_sse(resp):
    """Yield the JSON payloads of a Server-Sent Events stream."""
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- clients


class LlamaClient:
    engine = "llama.cpp"

    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def healthy(self) -> bool:
        try:
            return _get_json(self.base + "/health").get("status") == "ok"
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def meta(self) -> dict:
        """n_params and file size reported by the server (for bits per weight)."""
        try:
            data = _get_json(self.base + "/v1/models").get("data", [])
            return (data[0].get("meta") or {}) if data else {}
        except (urllib.error.URLError, OSError, ValueError):
            return {}

    def generate(self, prompt: str, max_tokens: int, ignore_eos: bool = True) -> Generation:
        payload = {
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0, "top_k": 1, "seed": 0,
            "stream": True, "cache_prompt": False, "ignore_eos": ignore_eos,
        }
        text, times, timings = [], [], {}
        t0 = time.perf_counter()
        with _post(self.base + "/completion", payload) as resp:
            for chunk in iter_sse(resp):
                if chunk.get("content"):
                    times.append(time.perf_counter() - t0)
                    text.append(chunk["content"])
                if chunk.get("stop"):
                    timings = chunk.get("timings") or {}
        wall = time.perf_counter() - t0
        return _build(text, times, wall, timings)

    def chat(self, messages: list[dict], max_tokens: int) -> tuple[str, int]:
        r = _post_json(self.base + "/v1/chat/completions",
                       {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 0})
        return r["choices"][0]["message"]["content"] or "", (r.get("usage") or {}).get("completion_tokens", 0)


class OpenAIClient:
    """Any OpenAI-compatible server (used for ``mlx_lm.server``)."""

    engine = "mlx"

    def __init__(self, base_url: str, model: str | None = None) -> None:
        self.base, self.model = base_url.rstrip("/"), model

    def healthy(self) -> bool:
        for path in ("/v1/models", "/health"):
            try:
                _get_json(self.base + path)
                return True
            except (urllib.error.URLError, OSError, ValueError):
                continue
        return False

    def meta(self) -> dict:
        return {}

    def _payload(self, **kw) -> dict:
        p = {"temperature": 0, **kw}
        if self.model:
            p["model"] = self.model
        return p

    def generate(self, prompt: str, max_tokens: int, ignore_eos: bool = True) -> Generation:
        text, times = [], []
        t0 = time.perf_counter()
        with _post(self.base + "/v1/completions", self._payload(prompt=prompt, max_tokens=max_tokens, stream=True)) as resp:
            for chunk in iter_sse(resp):
                choices = chunk.get("choices") or [{}]
                piece = choices[0].get("text") or ""
                if piece:
                    times.append(time.perf_counter() - t0)
                    text.append(piece)
        wall = time.perf_counter() - t0
        # Prompt length, asked only now: asked before, this call would put the exact prompt in
        # mlx_lm's prompt cache and the timed prefill would look ~10x faster than it is.
        usage = _post_json(self.base + "/v1/completions", self._payload(prompt=prompt, max_tokens=1)).get("usage") or {}
        n_prompt = int(usage.get("prompt_tokens") or 0)
        return _build(text, times, wall, {}, n_prompt=n_prompt)

    def chat(self, messages: list[dict], max_tokens: int) -> tuple[str, int]:
        r = _post_json(self.base + "/v1/chat/completions", self._payload(messages=messages, max_tokens=max_tokens))
        return r["choices"][0]["message"]["content"] or "", (r.get("usage") or {}).get("completion_tokens", 0)


def _build(text: list[str], times: list[float], wall: float, timings: dict, n_prompt: int = 0) -> Generation:
    ttft = times[0] if times else wall
    if timings.get("predicted_n"):
        # llama.cpp's own timings: exact token counts, no network jitter.
        pn, pms = timings.get("prompt_n", 0), timings.get("prompt_ms", 0)
        dn, dms = timings["predicted_n"], timings.get("predicted_ms", 0)
        return Generation(
            text="".join(text), prompt_tokens=pn, gen_tokens=dn, ttft_s=ttft, wall_s=wall,
            prefill_tps=pn / (pms / 1000) if pms else None,
            decode_tps=(dn - 1) / (dms / 1000) if dms and dn > 1 else None,
            timing_source="server", token_times=times,
        )
    n_gen = len(times)
    decode = (n_gen - 1) / (times[-1] - times[0]) if n_gen > 1 and times[-1] > times[0] else None
    return Generation(
        text="".join(text), prompt_tokens=n_prompt, gen_tokens=n_gen, ttft_s=ttft, wall_s=wall,
        prefill_tps=n_prompt / ttft if n_prompt and ttft else None, decode_tps=decode,
        timing_source="client", token_times=times,
    )


# --------------------------------------------------------------------------- servers


class Server:
    """A server process we start, own and stop."""

    def __init__(self, cmd: list[str], client, log_path: Path, startup_timeout: float = 600) -> None:
        self.cmd, self.client, self.log_path, self.timeout = cmd, client, log_path, startup_timeout
        self.proc: subprocess.Popen | None = None
        self.baseline_wired: int | None = None  # wired memory before the model is loaded (see memory.py)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        self.baseline_wired = wired_bytes()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early, see {self.log_path}")
            if self.client.healthy():
                return self
            time.sleep(0.5)
        self.__exit__()
        raise TimeoutError(f"server not ready after {self.timeout}s, see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
        if getattr(self, "_log", None):
            self._log.close()


def llama_server(model_path: Path, log_path: Path, ctx: int = 4096, port: int | None = None) -> Server:
    exe = shutil.which("llama-server")
    if not exe:
        raise RuntimeError("llama-server not found: brew install llama.cpp")
    port = port or free_port()
    cmd = [exe, "-m", str(model_path), "--port", str(port), "--host", "127.0.0.1",
           "-c", str(ctx), "-ngl", "999", "--parallel", "1"]
    return Server(cmd, LlamaClient(f"http://127.0.0.1:{port}"), log_path)


def mlx_server(repo: str, log_path: Path, port: int | None = None) -> Server:
    port = port or free_port()
    cmd = [sys.executable, "-m", "mlx_lm.server", "--model", repo, "--port", str(port), "--host", "127.0.0.1"]
    return Server(cmd, OpenAIClient(f"http://127.0.0.1:{port}", model=repo), log_path, startup_timeout=1800)
