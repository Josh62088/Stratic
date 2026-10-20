# This fork

A working fork of [Strata](https://github.com/Niko1221/Strata) kept as a **headless inference platform**: the
engine driven by a service, config files and an HTTP API, with no app or terminal around it, measured and
tuned against hardware we own.

Upstream is moving toward an integrated app. That is the right call for most users and the wrong shape for a
box that exists to serve a model to other software. So this fork tracks upstream's engine and diverges where a
headless deployment needs something different.

## What is here that upstream does not have

| path | what it is |
|---|---|
| `findings/` | A measurement study: a 125 B MoE on three consumer cards, with the method and the retractions published alongside the results |
| `findings/bug-reports/` | Engine reports, with reproduction steps and the evidence that separates a packer fault from a model fault |
| `tools/iq_pack.py` | Carries a fix: the expert predicate matched the 125 B's hyper-connection FFN tensors, so they were dropped from the pack and the engine died on the first token |

## The headline result

A 125 B MoE at **89.3 tok/s** with **807,866 tokens of context** on a $3,250 rig (3 x RTX 5060 Ti 16 GB,
Ryzen 9 5900XT, 128 GB DDR4-3600). Decode falls monotonically with precision across five quantisations, all
measured with a comparable VRAM expert cache:

| quantisation | bits/weight | decode |
|---|---|---|
| IQ2_XS | 2.06 | 89.3 tok/s |
| Q2_0 | 2.19 | 89.6 |
| IQ3_XXS | 3.06 | 83.6 |
| IQ3_S | 3.44 | 70.5 |
| UD-IQ4_XS | 4.19 | 51.9 |

Details, data and the three corrections the study had to make against itself: [`findings/`](findings/).

## Divergence from upstream

`stratic-qwen35-adapter` is **6 commits ahead of the `v0.1.42` tag and 3,191 behind `origin/main`** as of this
commit. It is not a mirror, and merging upstream is a deliberate act rather than a routine pull.

If you are looking for Strata itself, use [upstream](https://github.com/Niko1221/Strata). If you want the
measurements, or a headless deployment shape, this is the place.

## Licence

Unchanged from upstream. See `LICENSE`.
