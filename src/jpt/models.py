"""Benchmark configuration (TOML) and model downloads."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HF = "https://huggingface.co"


@dataclass
class Variant:
    model: str
    engine: str  # "llama.cpp" | "mlx"
    quant: str
    bits: float  # nominal bits per weight (16, 8, 4...)
    repo: str
    files: list[str] = field(default_factory=list)  # GGUF files (several if split)

    @property
    def id(self) -> str:
        return f"{self.model}/{self.engine}/{self.quant}"

    def local_path(self, models_dir: Path) -> Path | None:
        """First GGUF file (llama.cpp loads the other parts of a split model itself)."""
        if self.engine != "llama.cpp":
            return None
        return models_dir / self.repo.replace("/", "__") / self.files[0]

    def is_downloaded(self, models_dir: Path) -> bool:
        if self.engine != "llama.cpp":
            return True  # MLX models are fetched by mlx_lm into the Hugging Face cache
        d = models_dir / self.repo.replace("/", "__")
        return all((d / f).exists() for f in self.files)


@dataclass
class Settings:
    runs: int = 3
    max_tokens: int = 256
    idle_seconds: float = 5.0
    quiet_cpu_w: float = 1.5  # idle CPU power under which the machine is considered quiet
    cooldown_seconds: float = 20.0
    trial_pause_seconds: float = 0.0  # pause between trials (fanless Mac: limits throttling drift)
    cool_timeout_seconds: float = 600.0  # max wait for the thermal state to return to nominal
    gsm8k_n: int = 100
    gsm8k_max_tokens: int = 512
    perplexity_chunks: int = 40
    perplexity_ctx: int = 512
    ctx: int = 4096


@dataclass
class Config:
    settings: Settings
    variants: list[Variant]

    def select(self, pattern: str | None) -> list[Variant]:
        if not pattern:
            return self.variants
        return [v for v in self.variants if pattern in v.id]


def load_config(path: Path) -> Config:
    data = tomllib.loads(path.read_text())
    settings = Settings(**data.get("settings", {}))
    variants = []
    for m in data.get("model", []):
        for v in m.get("variant", []):
            variants.append(Variant(model=m["name"], engine=v["engine"], quant=v["quant"], bits=float(v["bits"]),
                                    repo=v["repo"], files=list(v.get("files", []))))
    return Config(settings, variants)


# --------------------------------------------------------------------------- downloads


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "joules-per-token"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != last and pct % 10 == 0:  # one line every 10 %, readable in a log file
                last = pct
                print(f"  {dest.name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB ({pct}%)", flush=True)
    tmp.rename(dest)


def fetch(variant: Variant, models_dir: Path) -> None:
    if variant.engine == "llama.cpp":
        d = models_dir / variant.repo.replace("/", "__")
        for f in variant.files:
            if not (d / f).exists():
                print(f"Downloading {variant.repo}/{f}")
                _download(f"{HF}/{variant.repo}/resolve/main/{f}", d / f)
    elif variant.engine == "mlx":
        print(f"Downloading (Hugging Face cache) {variant.repo}")
        code = f"from huggingface_hub import snapshot_download; snapshot_download({variant.repo!r})"
        subprocess.run([sys.executable, "-c", code], check=True)
    else:
        raise ValueError(f"unknown engine: {variant.engine}")


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2**30
