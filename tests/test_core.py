"""Unit tests — run with: python -m unittest discover -s tests (no Mac, model or network needed)."""

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jpt import bench, quality, report, stats, svg  # noqa: E402
from jpt.engine import LlamaClient, OpenAIClient  # noqa: E402
from jpt.models import load_config  # noqa: E402
from jpt.power import Energy, PowerMeter  # noqa: E402
from jpt.store import Store  # noqa: E402
from jpt.sweep import bits_per_weight  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeHandler(BaseHTTPRequestHandler):
    """Mimics llama-server and mlx_lm.server closely enough for the clients."""

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in events:
            self.wfile.write(f"data: {json.dumps(e) if not isinstance(e, str) else e}\n\n".encode())

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "m", "meta": {"n_params": 1_500_000_000, "size": 1_000_000_000}}]})
        self.send_error(404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/completion":
            n = req["n_predict"]
            ev = [{"content": "tok", "stop": False} for _ in range(n)]
            ev.append({"content": "", "stop": True, "timings": {
                "prompt_n": 50, "prompt_ms": 100.0, "predicted_n": n, "predicted_ms": n * 10.0}})
            return self._sse(ev)
        if self.path == "/v1/completions":
            if not req.get("stream"):
                return self._json({"choices": [{"text": "x"}], "usage": {"prompt_tokens": 42, "completion_tokens": 1}})
            return self._sse([{"choices": [{"text": "t"}]} for _ in range(req["max_tokens"])] + ["[DONE]"])
        if self.path == "/v1/chat/completions":
            return self._json({"choices": [{"message": {"content": "Step 1... so 3+4=7.\n#### 7"}}],
                               "usage": {"completion_tokens": 12}})
        self.send_error(404)


class FakeMeter(PowerMeter):
    name, available = "fake", True

    def energy_between(self, a, b):
        return Energy(seconds=b - a, joules={"cpu": 1.0, "gpu": 2.0, "ane": 0.0, "dram": 0.5})


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_llama_uses_server_timings(self):
        c = LlamaClient(self.url)
        self.assertTrue(c.healthy())
        g = c.generate("hi", 20)
        self.assertEqual((g.prompt_tokens, g.gen_tokens, g.timing_source), (50, 20, "server"))
        self.assertAlmostEqual(g.prefill_tps, 500.0)
        self.assertAlmostEqual(g.decode_tps, 19 / 0.2)
        self.assertEqual(bits_per_weight(c.meta()), 5.33)

    def test_openai_client_side_timings(self):
        g = OpenAIClient(self.url, model="m").generate("hi", 15)
        self.assertEqual((g.prompt_tokens, g.gen_tokens, g.timing_source), (42, 15, "client"))
        self.assertIsNotNone(g.decode_tps)

    def test_bench_rows_and_energy(self):
        rows = bench.run(LlamaClient(self.url), FakeMeter(), runs=2, max_tokens=10, idle_power_w=0.0, log=lambda *a: None)
        self.assertEqual(len(rows), len(bench.PROMPTS) * 2)
        short = rows[0]
        self.assertAlmostEqual(short["energy_j"], 3.5)
        self.assertAlmostEqual(short["j_per_token"], 0.35)

    def test_gsm8k_with_fake_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "gsm8k_test.jsonl").write_text("\n".join(json.dumps({"question": f"q{i}", "answer": f"... #### {7 if i % 2 == 0 else 8}"})
                                                         for i in range(10)))
            res = quality.run_gsm8k(OpenAIClient(self.url), d, n=10, max_tokens=64, meter=FakeMeter(), log=lambda *a: None)
            self.assertAlmostEqual(res["score"], 0.5)
            self.assertLess(res["ci_low"], 0.5)
            self.assertGreater(res["ci_high"], 0.5)


class PureTest(unittest.TestCase):
    def test_extract_answer(self):
        cases = {"so the total is\n#### 1,234": 1234, "The answer is $18.": 18, "x = 3, then 4.5": 4.5,
                 "\\boxed{42}": 42, "#### -5": -5, "no number": None}
        for text, want in cases.items():
            self.assertEqual(quality.extract_answer(text), want, text)
        self.assertTrue(quality.is_correct(18.0, "18"))
        self.assertFalse(quality.is_correct(None, "18"))

    def test_stats(self):
        self.assertEqual(stats.median([3, 1, 2]), 2)
        self.assertIsNone(stats.cv([1]))
        self.assertEqual(stats.stability_label(0.01), "stable")
        lo, hi = stats.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.404, places=2)
        self.assertAlmostEqual(stats.trapezoid_energy([(0, 10), (1, 10), (2, 20)]), 25)

    def test_pareto(self):
        rows = [
            {"variant": "a", "gsm8k": 0.6, "decode_tps": 50, "j_per_token": 0.5},
            {"variant": "b", "gsm8k": 0.5, "decode_tps": 80, "j_per_token": 0.3},
            {"variant": "c", "gsm8k": 0.5, "decode_tps": 40, "j_per_token": 0.6},  # dominated by a
            {"variant": "d", "gsm8k": None, "decode_tps": 99, "j_per_token": 0.1},
        ]
        self.assertEqual(report.pareto(rows), {"a", "b"})

    def test_configs_parse(self):
        for name in ("sweep.toml", "quick.toml"):
            cfg = load_config(ROOT / "configs" / name)
            self.assertTrue(cfg.variants)
            ids = [v.id for v in cfg.variants]
            self.assertEqual(len(ids), len(set(ids)))
            for v in cfg.variants:
                if v.engine == "llama.cpp":
                    self.assertTrue(v.files)

    def test_wait_quiet(self):
        class Noisy(PowerMeter):
            name, available = "noisy", True
            calls = 0

            def measure_idle(self, seconds=5.0):
                Noisy.calls += 1
                cpu = 4.0 if Noisy.calls < 3 else 0.5  # busy twice, then quiet
                return Energy(seconds=1.0, joules={"cpu": cpu, "gpu": 0.0, "ane": 0.0, "dram": 0.1})

        e = Noisy().wait_quiet(max_cpu_w=1.5, window_s=0, log=lambda *a: None)
        self.assertEqual(Noisy.calls, 3)
        self.assertAlmostEqual(e.joules["cpu"], 0.5)

    def test_vm_stat(self):
        from jpt.memory import parse_vm_stat
        text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1336.\nPages wired down:  204800.\n"
        self.assertEqual(parse_vm_stat(text), 204800 * 16384)
        self.assertIsNone(parse_vm_stat("nothing"))

    def test_svg(self):
        s = svg.line_chart([("a", [(4, 1), (8, 2)])], "t", "x", "y")
        self.assertTrue(s.startswith("<svg") and s.endswith("</svg>"))
        self.assertEqual(svg.line_chart([], "t", "x", "y"), "")
        for lo, hi in ((0.13, 0.195), (5, 16), (40.4, 65), (0, 1)):
            t = svg._nice_ticks(lo, hi)
            self.assertLessEqual(t[0], lo)
            self.assertGreaterEqual(t[-1], hi)

    def test_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            st = Store(Path(d) / "jpt.db")
            for q, bits, tps, j, acc in (("F16", 16, 30, 0.9, 0.6), ("Q4_K_M", 4, 70, 0.4, 0.55)):
                ctx = dict(session="s", engine="llama.cpp", model="m", quant=q, bits=bits, variant=f"m/llama.cpp/{q}")
                for p in ("short", "code", "long_ctx"):
                    st.add_run(**ctx, chip="M5", os="macOS", prompt_id=p, trial=1, decode_tps=tps, prefill_tps=500,
                               ttft_ms=100, j_per_token=j, j_per_token_dyn=j / 2, peak_mem_mb=2048, bpw=bits + 0.5)
                st.add_quality(**ctx, task="gsm8k", n=100, score=acc, ci_low=acc - 0.1, ci_high=acc + 0.1, details=[])
                st.add_quality(**ctx, task="perplexity", n=10, score=10 + 16 / bits, ci_low=9, ci_high=12)
            st.add_thermal(session="s", variant="m/llama.cpp/Q4_K_M", t_s=60, decode_tps=70, avg_power_w=8, thermal_state=0)
            path = report.build(st, "s", Path(d) / "out")
            st.close()
            text = path.read_text()
            self.assertIn("m/llama.cpp/Q4_K_M", text)
            self.assertIn("★", text)
            for f in ("gsm8k.svg", "decode.svg", "energy.svg", "pareto.svg", "perplexity.svg", "thermal.svg", "summary.csv"):
                self.assertTrue((Path(d) / "out" / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
