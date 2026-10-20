# Findings

Measurements taken on a three-GPU consumer rig running this engine, with the method published alongside the
result so that every number here can be checked, and every wrong turn is visible.

The study: [inference-optimization.md](inference-optimization.md)

## The headline

A 125-billion-parameter MoE model serves **89.3 tok/s** and holds **807,866 tokens of context** on a $3,250
consumer box — 3 x RTX 5060 Ti 16 GB, Ryzen 9 5900XT, 128 GB DDR4-3600.

Decode falls monotonically with precision, measured on one rig with every quantisation given a comparable VRAM
expert cache:

| quantisation | bits/weight | decode, mean of 3 cold repeats |
|---|---|---|
| IQ2_XS | 2.06 | 89.3 tok/s |
| Q2_0 | 2.19 | 89.6 |
| IQ3_XXS | 3.06 | 83.6 |
| IQ3_S | 3.44 | 70.5 |
| UD-IQ4_XS | 4.19 | 51.9 |

## What the data directory contains

Every JSON here is the raw output of a harness in this study, including the measurements that turned out to be
**wrong**. That is deliberate: the study retracts three of its own conclusions, and a retraction whose data has
been deleted cannot be verified.

| file | what it is |
|---|---|
| `quant-curve-corrected.json` | The five-quantisation curve, corrected harness: 2 warm-ups discarded, 3 repeats per depth, shuffled order, medians. This is the source of the paper's quantisation table. |
| `ud-iq4-xs-depth-sweep.json` | UD-IQ4_XS depth sweep 2K-262K, pre-correction harness. |
| `ud-iq4-xs-extended-807k.json` | The 807,866-token cold prompt on a 1,048,576-token window (yaRN x4). |
| `ud-iq4-xs-warmup-effect.json` | The run that isolated the warm-up bias: first request after load vs the rest. |
| `run-to-run-noise-floor.json` | Identical config measured twice: 5.5 tok/s mean delta, 19 tok/s on one depth. This number sets the resolution limit of everything else. |
| `armd-depth-sweep.json` | Setup B, IQ2_XS, decode against context depth (pre-correction). |
| `armd-resident-128gb.json` | The 128 GB resident configuration, 10/10 stable, `files 0 blobs`. |
| `armc-working-set.json` | Working-set sizing experiments. |
| `arm1-32gb-2133.json` | Setup A baseline at 32 GB, XMP off. |
| `armbc-trials.json` | XMP and concurrency trials. |
| `armb2-context-use.json` | Prefix-cache and context-reuse measurements. |
| `quant-curve-precorrection.json` | The pre-correction five-quant curve. Kept deliberately: it is the wrong result the study retracts, and a retraction without the retracted data is not checkable. |
| `iq3-xxs-precorrection.json` | IQ3_XXS pre-correction, including the 1.58 GiB cache starvation that produced the false anomaly. |
| `iq3-s-precorrection.json` | IQ3_S pre-correction, taken with --expert-cache auto at zero slots. |
| `q2-0-precorrection.json` | Q2_0 pre-correction. |
| `cuda13-build-test.json` | The CUDA 13 build verification. |

## Three corrections this study had to make

1. **Harness: the first request after engine start is slow.** 50.9 tok/s against 56.6 for the second. Every
   early figure in this study was measured with that request first, understating the whole curve by ~9%.

2. **Harness: a decode rate over ~49 tokens measures overhead, not rate.** The study already carried the rule
   *"a rate over a small count measures overhead, not rate"* and then violated it for decode.

3. **`--expert-cache auto` silently starves the cache.** On a ~60 GB pack it sized itself from a momentary
   free-VRAM reading and chose **zero slots**, leaving 6 GB of a 16 GB card idle and reporting 57 tok/s for a
   quantisation that does 70.5. This produced a false "precision does not predict speed" result that was
   retracted before publication.

## Reproducing

The harness rules that produced the corrected table:

- two discarded warm-up requests after each engine load
- ~250 words generated per request, so the decode rate reaches steady state
- three repeats per depth, median reported, never a single sample
- depth order shuffled per repeat, so no depth is systematically first
- every prompt carries a unique nonce and `reused 0` is verified, so nothing is a cache hit
- the serving model id is asserted before each leg, because `/v1/models` returning 200 does not mean the
  engine you started is the engine that answered

## Caveats

- One rig, one model family, one driver. Nothing here is a universal constant.
- The 262 K and 807 K measurements predate the harness correction and are understated by roughly a quarter;
  they are kept for the curve's shape, which does not depend on the offset.
- `UD-Q4_K_XL` is not measured: at 111 GB of model it needs more NVMe than this rig has free, so it could
  only run disk-bound, which measures the disk.
