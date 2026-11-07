# Methodology

This document describes how each number in `results/` is produced, and the choices behind it.

## Hardware and software

| | |
|---|---|
| Machine | MacBook Air, Apple M5, 16 GB unified memory, passive cooling |
| Engines | llama.cpp (`llama-server`, Homebrew build), MLX (`mlx_lm.server`) |
| Models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, official GGUF and MLX community conversions |

Chip, OS version, engine version, model file and quantization are stored with every measurement (`results/jpt.db`, SQLite).

## Configurations

Each model is run in 10 configurations: 7 GGUF quantization levels with llama.cpp (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K) and 3 MLX levels (bf16, 8-bit, 4-bit). The effective compression is reported as **bits per weight**, computed as `file size × 8 / parameter count`. K-quant labels understate the real size, because some tensors (embeddings, output layer) are kept at higher precision; Q4_K_M is about 4.9–5.0 bits per weight.

## Throughput

Three prompts cover different regimes: a long-form explanation, a code-generation task (both 256 output tokens) and a summary of a long input (64 output tokens), which weights prefill more heavily.

For each configuration:

1. The server is started and a warm-up request is sent and discarded.
2. Each prompt is run 3 times with greedy decoding (`temperature = 0`) and a fixed output length, with a 10 s pause between trials. Before each configuration, the run waits (up to 10 min) for the macOS thermal state to return to nominal, so no configuration inherits heat from the previous one.
3. Prompt caching is defeated, so prefill is paid on every trial: llama.cpp is called with `cache_prompt: false`; for MLX, whose server reuses the longest cached prefix, every request starts with a unique tag so that no prefix is shared.

Reported values are the **median** over trials:

- **Decode throughput** (tokens/s) during generation. Decoding reads every weight once per token, so it is bound by memory bandwidth, which is why it scales with compression.
- **Prefill throughput** (tokens/s) and **time to first token**. Prefill processes the prompt in parallel and is compute bound.

The coefficient of variation is reported alongside: below 5% the result is labelled *stable*, below 10% *variable*, above *unstable*.

llama.cpp timings come from the server's own `timings` field (native `/completion` endpoint). MLX does not report them, so they are taken from client-side streaming timestamps.

## Energy

Energy is read from Apple's private **IOReport** library, *Energy Model* group, through `ctypes`. The channels are cumulative energy counters per block (CPU, GPU, ANE, DRAM), so the energy of a request is the difference between two snapshots taken just before and just after it. There is no sampling interval and no integration error, and no `sudo` is required. `powermetrics` is available as a fallback.

- **J/token** = total SoC energy of the request / generated tokens.
- **Idle-subtracted J/token** removes the idle power, measured for 5 s before each configuration, multiplied by the request duration. This isolates the cost of inference from the base draw of the machine.

To limit noise, each configuration waits until idle CPU power is below 1.5 W (background indexing, updates) and is followed by a 30 s cool-down. The figures include the whole SoC: they are directly comparable between configurations on the same machine, not with GPU board power on other platforms.

## Memory

With Metal, model weights are allocated in wired GPU memory that is not attributed to the server process, so the process resident size alone under-reports usage by an order of magnitude. Peak memory is measured as:

`(system wired memory during the request − wired memory before the server started) + process physical footprint`

Wired memory is read from `vm_stat`, the footprint from `proc_pid_rusage` (`ri_phys_footprint`), sampled during each request.

## Thermal behaviour

`jpt thermal` generates continuously for a set duration (15 min in the study) and records throughput, power and the macOS thermal state (`NSProcessInfo.thermalState`) over time. This captures throttling, which short benchmarks miss on a fanless machine.

## Quality

**GSM8K.** The first 300 problems of the test split, sent through the chat endpoint of each engine with the model's chat template. The model is asked to give its final answer as `#### <number>`. The answer is extracted from that marker, with fallbacks to "the answer is N", `\boxed{N}` or the last number, and compared numerically to the reference. Accuracy is reported with a **95% Wilson score interval**: about ±4 to 5 points at n = 300. Differences smaller than the interval are not interpreted.

**Perplexity.** WikiText-2 (raw) test set, measured with `llama-perplexity` on 40 chunks of 512 tokens. Perplexity measures how well the model predicts held-out text and is more sensitive than GSM8K to small degradations. It is computed for llama.cpp configurations only.

**KL divergence.** `llama-perplexity` first saves the F16 model's full next-token distributions on the same 40 WikiText-2 chunks, then compares each quantized model against them. Two figures are reported: the mean KL divergence between the two distributions, and the share of positions where both models rank the same token first. Unlike GSM8K, which only checks final answers, this measures how far the compressed model drifts from the original at every token, with tight intervals.

## Selection

A configuration **dominates** another if it is at least as good on GSM8K accuracy, decode throughput and energy per token, and strictly better on at least one. The **Pareto frontier** is the set of non-dominated configurations; it is marked ★ in the report.

## Reproducibility

- Configurations are declared in TOML (`configs/sweep.toml`).
- Sweeps are resumable: a configuration already measured in a session is skipped.
- `jpt report` regenerates every table and chart from the database.
- Unit tests run against a fake inference server and require no model.

## Known issues in the current results

- **Two unstable configurations.** Qwen2.5-1.5B Q3_K_M and Q2_K still exceed 10% CV on decode throughput despite the pauses and the thermal gate.
- **MLX memory is likely overestimated.** The wired-memory delta also includes MLX's buffer cache, which is not freed between requests.

## Changes from the first run

The first run had MLX prefill and time-to-first-token about 10× too fast (a request sent just before the timed one warmed MLX's prompt cache) and several unstable 1.5B throughput figures (heat carried over between trials). Both were fixed (unique request prefix; pauses between trials and a thermal gate) and every speed, memory and energy figure was re-measured.
