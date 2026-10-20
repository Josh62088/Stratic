# `dequant launch: an illegal memory access` survives every memory tier

A CUDA fault in the dequantisation/prefill path on sm_120 that is independent of RAM size, of the expert
tier in use, and of available VRAM. It reproduces across two very different configurations of the same
machine, which is what makes it worth reporting as one bug rather than two.

## Version

    checkout   : v0.1.42-6-g36fe6e9
    engine     : 0.1.42, sm_120 build (locally installed prebuilt, CUDA 12.8)
    model      : Qwen3.8-Flash-Next IQ2_XS
    hardware   : 3 x RTX 5060 Ti 16 GB (sm_120), Gigabyte B550, Ubuntu 24.04, kernel 6.8
    driver     : 595.84

## Two configurations, one fault

| | **A — file tier** | **B — resident RAM** |
|---|---|---|
| CPU | Ryzen 5 5500 (6C/12T) | Ryzen 9 5900XT (16C/32T) |
| RAM | 32 GB DDR4-2133 | **128 GB DDR4-3600** |
| expert mode | `--mmap-experts` | `--resident-experts` |
| experts in RAM | **0** — read from NVMe per token | **28.61 GiB, page-locked** |
| file tier | active: 1,145 MB read per request, 2,492 major faults | **`files 0 blobs`** |
| context | 32768 | 131072 |
| GPU | 1 card visible (BIOS BAR fault) | 1 card, GPU 2, x8 Gen4 |
| decoded (short prompt) | **23.8 – 39.8 tok/s** | **~103 tok/s** (30 tok / 0.29 s) |
| survived repeated requests | **2 – 4, then exit 1** | **10 / 10** |
| **died on a ~2,000-token prompt** | yes | yes |

**Configuration B is the one that matters for this report:** it is the configuration the engine
recommends, it achieves the intended `resident RAM mode ... (no file reads)`, it is dramatically more
stable on short prompts — **and it still dies with the same message.**

## The fault

    strata serve: 1724 MiB of VRAM free with everything loaded
    dequant launch: an illegal memory access was encountered

In configuration A the analogous line was:

    dequant launch: an illegal memory access was encountered

and in a later prefill-path variant, briefly:

    prefill copy_i32: an illegal memory access was encountered

so there is at least one related site in the prompt path as well as the dequant path.

## What has been ruled out

- **Not RAM pressure.** B has 125 GiB total with 120 GiB available; the engine takes 28.61 GiB
  page-locked and reports `no file reads`. No OOM-killer event appears in `dmesg` or the journal.
- **Not VRAM exhaustion.** At the crash, `1724 MiB` (A: `380 MiB`) of VRAM was free. Raising the reserve
  with `--vram-reserve-mib 2048` moved the free figure from 380 to 1724 MiB and did not change the
  outcome.
- **Not the file tier.** B has `files 0 blobs` — zero blob reads from the file — and fails identically.
- **Not prompt-specific.** A 55-token prompt succeeds ten times in a row in B; the fault appears as the
  prompt grows. In B, 200 / 500 / 900 / 2000-token prompts all failed, so the threshold is somewhere
  between 55 and 200 tokens.
- **Not fixed by `STRATA_PREFILL_CPU_SHARE=0`.** Tried; still fails.

## Kernel-side evidence (from the earlier file-tier configuration)

    NVRM: Xid (PCI:0000:01:00): 31, pid=125865, name=strata, channel 0x00000003, intr 00000000.
          MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_11 faulted @ 0x9841_de9d4000.
          Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ

Three occurrences with different PIDs, different GPCs and different clients. `ENGINE GRAPHICS` rather
than a compute engine, and the fault type is a page-directory-entry fault on a virtual **read** — the
mapping did not exist, rather than the address being out of range.

## How to reproduce

    ./setup.sh --setup --model IQ2_XS --family qwen --context 131072 --vision no --yes --no-start \
        --low-ram resident --gpu 2 --vram-reserve-mib 2048
    ./run-iq2_xs.sh
    # with a ~2000-token prompt:
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"x","messages":[{"role":"user","content":"<2000 tokens>"}],"max_tokens":48}'

The engine loads, fills its 6202-slot GPU expert cache, completes the page-locked 28.61 GiB copy, reports
`resident RAM mode ... (no file reads)`, and then exits on the first substantial prompt. The server
restarts it; the next request repeats.

## Suggested lines of inquiry

1. **`ENGINE GRAPHICS` in the Xid is the lead.** A CUDA kernel launch normally faults on a compute
   channel. If the dequant or prompt-copy path is dispatched on the graphics queue, or if a mapped region
   is unmapped while the launch still holds a pointer into it, this is the expected signature.
2. **The prompt path borrows CUDA0 cache slots** (`the prompt path borrows 2925 CUDA0 cache slots
   (3.90 GiB)`). A borrow that assumes more headroom than exists would fault as the prompt grows; the
   fact that the observed minimum prompt size to fail is small suggests the borrow accounting is worth
   checking rather than the size of the allocation.
3. **`prefill copy_i32` and `dequant launch` may share a cause.** Both fail on a virtual read with the
   mapping absent.

## What would help most from the maintainers

Whether `dequant` and the prompt copy are supposed to run on a compute queue on sm_120, and whether
`--resident-experts` changes which queue they are dispatched to. Configuration B exercises a path
(`resident RAM mode`, page-locked complement) that the file-tier configuration does not, and it fails at
a lower prompt size — which may narrow it.
