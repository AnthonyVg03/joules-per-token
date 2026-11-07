# Is 4-bit quantization free? I measured it on a fanless MacBook Air

Everyone running LLMs locally quantizes. The folklore says 4-bit is "basically free" in quality and saves a lot of energy. I wanted numbers, so I built a small benchmark tool and measured 20 configurations of Qwen2.5 (1.5B and 3B) on a MacBook Air M5 with 16 GB and no fan: llama.cpp from F16 down to Q2_K, and MLX at bf16, 8-bit and 4-bit.

For each configuration I measured generation speed, prompt processing speed, peak memory, energy per token (read from Apple's hardware energy counters), accuracy on GSM8K, perplexity and KL divergence from the F16 model, plus 15 minutes of sustained load.

## 1. Down to 4 bits, the quality loss is real but invisible on a task

On 300 GSM8K problems, Qwen2.5-3B scores 85.0% in F16 and 84.7% in Q4_K_M. Q8, Q6 and Q5 land at 84.3%. Statistically, these are the same model.

That does not mean nothing changes. Q4_K_M picks the same next token as F16 91% of the time. The model drifts, but not in ways that change final answers. Below 4 bits the drift accelerates: 85% agreement at Q3, 70% at Q2, and GSM8K accuracy follows (80%, then 60%).

## 2. Prefer a bigger model, compressed harder

The 1.5B model collapses much faster than the 3B: at Q3_K_M it falls to 49% on GSM8K, against 80% for the 3B. And the 3B at Q3 (2.2 GB, 79.7%) beats the 1.5B at full precision (3.4 GB, 74.7%). With a fixed memory budget, take the largest model that fits after quantization.

## 3. 2.5× faster does not mean 2.5× cheaper

From F16 to Q4_K_M, the 3B model goes from 21 to 53 tokens per second. Energy per token only drops by 28%.

The reason: generating a token means reading every weight from memory once. Smaller weights are read faster, so speed scales with compression. But the chip is now working harder, and its power draw rises from 7.8 W to 13.4 W. Energy per token is power divided by speed, so most of the speed gain is eaten by the extra power.

## 4. The engine matters as much as the bit count

At 4 bits, Apple's MLX uses 28% less energy per token than llama.cpp at the same speed. But its simpler 4-bit scheme costs about 10 GSM8K points (74% vs 85%). llama.cpp's K-quants keep more precision where the model is sensitive, and it shows.

## 5. A 30-second benchmark lies on a fanless laptop

After 15 minutes of continuous generation, Q4_K_M slows from 54 to 45 tokens per second (−16%), F16 by 10%. The faster configuration draws more power, heats more, and is throttled more.

## What I would use

Qwen2.5-3B in Q4_K_M with llama.cpp: F16-level accuracy, 2.5× the speed, 2.5 GB of memory.

## Method notes and limits

- Medians over repeated trials, a pause between trials, and a cool-down gate before each configuration. Energy covers the whole chip, with idle power subtracted.
- One machine, one model family, two small sizes. The next step is a 7–8B model and a second family.
- Full results: [`results/final/report.md`](../results/final/report.md). Methodology: [`METHODOLOGY.md`](METHODOLOGY.md).
