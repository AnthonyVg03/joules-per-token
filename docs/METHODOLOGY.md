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
2. Each prompt is run 3 times with greedy decoding (`temperature = 0`) and a fixed output length.
3. Prompt caching is disabled, so prefill is paid on every trial.

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

**GSM8K.** The first 100 problems of the test split, sent through the chat endpoint of each engine with the model's chat template. The model is asked to give its final answer as `#### <number>`. The answer is extracted from that marker, with fallbacks to "the answer is N", `\boxed{N}` or the last number, and compared numerically to the reference. Accuracy is reported with a **95% Wilson score interval**: at n = 100 the interval is about ±8 to 10 points, and differences smaller than that are not interpreted.

**Perplexity.** WikiText-2 (raw) test set, measured with `llama-perplexity` on 40 chunks of 512 tokens. Perplexity measures how well the model predicts held-out text and is more sensitive than GSM8K to small degradations. It is computed for llama.cpp configurations only.

## Selection

A configuration **dominates** another if it is at least as good on GSM8K accuracy, decode throughput and energy per token, and strictly better on at least one. The **Pareto frontier** is the set of non-dominated configurations; it is marked ★ in the report.

## Reproducibility

- Configurations are declared in TOML (`configs/sweep.toml`).
- Sweeps are resumable: a configuration already measured in a session is skipped.
- `jpt report` regenerates every table and chart from the database.
- Unit tests run against a fake inference server and require no model.

## Known issues in the current results

- **MLX prefill and time to first token are invalid.** A request used to read the prompt length was sent before the timed request and populated MLX's prompt cache, making prefill appear about 10× faster. The request is now sent after timing; these figures will be re-measured. Decode throughput, energy and accuracy are not affected.
- **Unstable 1.5B llama.cpp throughput.** Several runs exceed 10% CV, attributed to thermal state carried over from previous trials. A pause between trials and a gate that waits for the nominal thermal state before each configuration are now implemented.
- **MLX memory is likely overestimated.** The wired-memory delta also includes MLX's buffer cache, which is not freed between requests.
