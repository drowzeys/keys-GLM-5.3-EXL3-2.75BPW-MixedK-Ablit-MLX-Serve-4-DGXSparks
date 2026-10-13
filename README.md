# keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks

# Built on [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu and a third-party inference engine (1.0.7, see [NOTICE.md](NOTICE.md))

The full **GLM-5.3** (753B parameters, EXL3 2.75 bpw mixed-K, abliterated) served across **four NVIDIA DGX Sparks**
(GB10, one rank a Spark over ConnectX-7 RoCE, tensor parallel 4). mlx-serve is the HTTP server on rank 0; the inference
engine does the forward pass, drafting and sampling on all four. One prebuilt image, drafter included.

| | |
|---|---|
| Image | `ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-4-dgxsparks:2026-10-12-r2` (= `latest`) |
| Weights | [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated) (gated, automatic approval) |
| Drafter (in the image) | [drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2), GLM-5.3 License |
| Endpoint | `http://<rank 0>:18992/v1`, OpenAI- and Anthropic-compatible, model name `model`, no authentication |
| Measured configurations | 32K context × 4 streams (default), 64K × 2 (`PRESET=64k`), 250K × 2 (`PRESET=250k`), 1M × 1 (`PRESET=1m`) |

## Performance

Four DGX Sparks, the published image started with `one-shot.sh`, default settings. GLM-5.3 is a reasoning model, so
every number is with thinking on (the server's default, High effort).

| Measurement | Result |
|---|---|
| Prose, one stream (`./one-shot.sh bench`: greedy, 512 tokens) | **38.3 tok/s** |
| Code, one stream (same) | **56.3 tok/s** |
| 25K-token prompt, first time (prefill + answer) | 23.2 s |
| The same 25K prompt with a changed last line (prompt reuse) | **3.3 s** |
| Load time, `up` to serving | 352 s |
| Memory left on each node while serving | 9-10 GB |

Speeds vary by about 9 % from one load to the next on this cluster.

### Other context sizes

`PRESET=64k ./one-shot.sh up` and `PRESET=1m ./one-shot.sh up` use a second kernel pack in the image, and so does
`PRESET=250k` (two streams of 250,000 tokens, [below](#image-2026-10-12-r2-two-long-context-streams-preset250k)).

| | Default | `PRESET=64k` | `PRESET=1m` |
|---|---|---|---|
| Context × streams | 32,768 × 4 | 65,536 × 2 | 1,000,000 × 1 |
| Prose, one stream | 38.3 tok/s | 36.8 tok/s | 35.0 tok/s |
| Code, one stream | 56.3 tok/s | 57.2 tok/s | 44.7 tok/s |
| 25K-token prompt, first time | 23.2 s | not measured | 61.5 s |
| The same prompt with a changed last line | 3.3 s | not measured | 7.7 s |
| Memory left on each node | 9-10 GB | 12-14 GB | 8-10 GB |
| Load time | 352 s | 318 s | 318 s |

- **`PRESET=1m`** splits the cache across the four Sparks (decode context parallel), which is what makes 1M fit. It
  turns off the dense copy of the MTP layer (`TF_GLM53_MTP_DENSE=0`, 5 GB a Spark) and caps the kept prompt prefixes at
  4 GB (`TF_GLM53_CACHE_GIB=4`). Measured from image `2026-10-10-r4`: `evidence/2026-10-10/r4-*.log`.
- **`PRESET=64k`** was measured on the previous image tag with this kernel pack mounted from disk, not started from
  this tag.

## Image `2026-10-12-r2`: two long-context streams (`PRESET=250k`)

Built on [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu, a third-party inference engine (see
[NOTICE.md](NOTICE.md)), [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (the EXL3 format and
quantizer) and GLM-5.3 by [Z.ai](https://huggingface.co/zai-org).

`2026-10-12-r2` is `2026-10-12` with one addition: **two streams of 250,000 tokens each**. Until now more than one
stream meant every stream kept its whole cache on every Spark, which is why the shapes stopped at 32K × 4 and
64K × 2. The new preset splits the cache across the four Sparks as `PRESET=1m` does and gives each stream its own rows
of it.

```sh
for n in $NODES; do ssh $n docker pull ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-4-dgxsparks:2026-10-12-r2; done
PRESET=250k ./one-shot.sh up        # 250,000 tokens × 2 streams
```

Measured from the image as published, these weights, four DGX Sparks, thinking on, greedy, 512 tokens
(`evidence/2026-10-12-r2/`):

| | `PRESET=250k` (250,000 × 2) | `PRESET=1m` (1,000,000 × 1), same image |
|---|---|---|
| Prose, one request | 34.7 tok/s | 33.1 and 33.9 tok/s (two runs of the bench) |
| Code, one request | 46.0 tok/s | 44.1 and 44.6 tok/s |
| Prose, two requests at once: each / together | 22.8 and 24.0 / **46.4 tok/s** | one at a time |
| Code, two requests at once: each / together | 33.0 and 26.6 / **53.0 tok/s** | one at a time |
| Two 58K-token prompts at once, a different passphrase at half depth in each | both found; 117,026 prompt tokens in 312 s (375 tok/s in all) | not run on this tag |
| 25K-token prompt, first time / changed last line | 60.5 s / 7.6 s | 61.3 s / 7.7 s |
| Replies with two requests at once against one at a time (8 prompts, 41 to 18,840 tokens) | 8 of 8 identical | |
| Checks (`bench/checks.py`) | 27 replies, no errors | 27 replies, no errors |
| Memory left on each node | 12 GB | 9 GB |
| Load time | 335 s | 317 s |

- **Measured on the unmodified 2.75 bpw pack with the same image build:** two 106K-token prompts at once, both
  passphrases found, 213,023 prompt tokens in 575 s; prose 36.0 tok/s alone and 47.1 together; code 48.6 alone and
  47.7 together (no gain on that run).
- **What it costs:** as in `PRESET=1m`, prompts are read at about 370 tok/s, about 2.5 times slower than the default
  shape, and two prompts being read at once share that rate. One request alone decodes at the `1m` rate, not the
  default shape's. The dense copy of the MTP layer is off, as in `1m`.
- **Two streams, not more.** Four streams with the cache split ran a node out of memory while loading in our tests;
  the engine now refuses those shapes when it starts. 262,144 × 2 served correctly on the unmodified pack but left
  8 GB on a node against 12 GB here, so the launcher offers `250k` only.
- **`PRESET=1m` on this tag read 3 to 5 % lower than on `2026-10-12` earlier the same day** (35.2 and 46.0 tok/s
  there), with identical benchmark texts. On the unmodified pack the two tags gave the same `1m` speed (36.2 / 48.4
  against 36.0 / 48.4), and speeds on this cluster vary by several percent from one load to the next; each shape was
  loaded once.
- **Limits:** measured with the DSpark drafter only. Prompts beyond 58K tokens a stream were not tried on these
  weights. The default 32K × 4 and `64k` shapes were not started from this tag with these weights.

## Image `2026-10-12`: a 6-bit output head and a lighter health check

Built on [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu, a third-party inference engine (see
[NOTICE.md](NOTICE.md)), [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (the EXL3 format and
quantizer) and GLM-5.3 by [Z.ai](https://huggingface.co/zai-org).

| Change | Effect |
|---|---|
| The output head is read as a 6-bit EXL3 linear when the checkpoint has one (`lm_head-6bit/lm_head-6bit.safetensors`, 714 MB, now in the weights repository) | A head launch takes 0.79 ms on a Spark against 2.16 ms for the bf16 head. Measured on the unmodified 2.75 bpw pack, whose output head is byte-identical to this checkpoint's: as-served KL against the BF16 checkpoint 0.175612 with the bf16 head, 0.175724 with this one, over 137,490 positions; top-1 agreement 89.64% and 89.60%. Some greedy replies differ |
| The per-window health check no longer does its own all-gather (`TF_GLM53_ROCE_HEALTH_LOCAL=1`) | Replies identical to the previous build; about 1% faster |

Measured from each image as published, same day, four DGX Sparks, thinking on, greedy, 512 tokens
(`evidence/2026-10-12/`):

| | `2026-10-11` | `2026-10-12` |
|---|---|---|
| Prose, `PRESET=1m`, these weights | 34.9 tok/s | 35.2 tok/s |
| Code, `PRESET=1m`, these weights | 44.4 tok/s | **46.0 tok/s** |
| 25K-token prompt, first time / changed last line, `PRESET=1m`, these weights | 62.0 s / 7.7 s (10-11) | 62.2 s / 7.8 s |
| 106K-token prompt, passphrase at half depth, `PRESET=1m`, these weights | not run | found in 287 s; prefill 371 tok/s |
| Prose / code, one stream, 32K × 4, unmodified pack | 37.5 / 55.8 tok/s | 39.6 / 57.8 tok/s |
| Prose, all streams together: 1 / 2 / 4 at once, 32K × 4, unmodified pack | 36.8 / 51.6 / 66.6 tok/s | 38.6 / 53.6 / 68.5 tok/s |
| Code, all streams together: 1 / 2 / 4 at once, 32K × 4, unmodified pack | 54.1 / 60.5 / 77.4 tok/s | 55.8 / 62.5 / 80.5 tok/s |

- **Checks** (`bench/checks.py`): 27 replies and no errors in `PRESET=1m` on these weights.
- **Not re-measured on this tag with these weights:** the default 32K × 4 and `64k` shapes. The 32K × 4 rows above ran
  on the unmodified pack with the same engine build; the tables in Performance are from `2026-10-10-r4`.
- **Turning the head off:** `HEAD_Q=off ./one-shot.sh up` serves with the bf16 head, as `2026-10-11` did. Without
  the head file the image logs `no /model/lm_head-6bit/lm_head-6bit.safetensors: bf16 output head` and does the same.
- **Limits:** the 6-bit head was measured with the DSpark drafter only.

## Checks the image passed

Run with `HOST=<rank 0> python3 bench/checks.py`; logs in `evidence/2026-10-10/`. The last three rows ran on the same
binaries just before they were packaged (`long-replies.log`); the rest ran on the image.

| Check | Result |
|---|---|
| Short answers; `max_tokens=1` | Stops on its own; the limit is honoured |
| Client disconnects 0.05 s, 0.5 s and 3 s into a stream | The next request is answered in under 1 s (default) and in 0.2 to 0.7 s (`PRESET=1m`) |
| A title request beside a dropped stream | Both answered |
| 2 dropped streams beside 2 surviving, then 4 at once | All finish on their own |
| Sampled replies (T = 0.8, two seeds) | Clean stop at 440 and 400 tokens |
| Tool call | Correct function and arguments |
| 7 short prompts at once on 4 streams | All finish on their own |
| Code question | Stops at 1,083 tokens |
| Long essay with a 6,000-token limit | Reached the limit while still writing new text (no repetition) |

## Quick start

### One script (new clusters)

On a machine with key-based ssh to the four Sparks (one of them is fine):

```sh
git clone https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks && cd keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks
HF_TOKEN=<your token> ./install.sh <spark1> <spark2> <spark3> <spark4>   # the four fabric addresses, rank 0 first
./one-shot.sh up && ./one-shot.sh bench
```

`install.sh` checks each Spark (ssh, GB10 and memory, Docker with GPU access, the RoCE devices, disk space), applies
the two kernel settings a rank needs, pulls the image and checks its digest, downloads the checkpoint and the 6-bit
output head to `~/models/GLM-5.3-EXL3-2.75BPW-Abliterated` on each Spark (resumable; `VERIFY=1` also checks every
file's sha256), writes `my-cluster.env`, which `one-shot.sh` reads, and runs the start check.

| Setting | Meaning |
|---|---|
| `HF_TOKEN` | The weights repository is gated with automatic approval: accept its terms on huggingface.co first. Without the variable, the token `hf auth login` saved on each Spark is used |
| `MODEL=<directory>` | Where the checkpoint goes, the same path on all four |
| `SHARED=1` | `MODEL` is one shared directory all four mount: download once (`SHARED_HOST=<address>` names the machine that can write to it) |
| `SKIP_WEIGHTS=1` | The checkpoint is already at `MODEL` |

- **The container image is still private** as of 2026-10-12, so the pull works only for accounts with access until
  the package is made public.
- **Tested here:** the installer's checks, image step and settings file ran against our four Sparks, and the download
  path ran on the repository's small files with a token. A full download through it and a start from a directory it
  filled were run for the stock checkpoint's recipe, which shares this script, not for this one.

### By hand

On a machine with passwordless ssh to the four Sparks:

```sh
git clone https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks && cd keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-4-DGXSparks
export NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"      # fabric addresses, rank 0 first
export MODEL=/models/keys-GLM-5.3-EXL3-2.75BPW-Abliterated  # the same path on all four

hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated --local-dir $MODEL    # once, ~250 GB, on every node or a share
# already have the checkpoint? fetch the 6-bit output head only (714 MB): add  --include "lm_head-6bit/*"
for n in $NODES; do ssh $n docker pull ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-4-dgxsparks:2026-10-12-r2; done
for n in $NODES; do scp node/gb10-node-settings.sh $n:/tmp/ && ssh -t $n sudo /tmp/gb10-node-settings.sh; done   # after every reboot

./one-shot.sh check     # must end with "check: OK"
./one-shot.sh up        # about 6 minutes; prints SERVING and the address
./one-shot.sh bench     # expect about 38 tok/s prose, 56 tok/s code
./one-shot.sh down
```

```sh
curl http://<rank 0>:18992/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

## Requirements

| | |
|---|---|
| Nodes | 4 × NVIDIA DGX Spark (GB10, 128 GB unified memory), nothing else on their GPUs |
| Fabric | ConnectX-7 RoCE between all four (a switch or direct cables); one or two rails |
| Free memory | 100 GB available on every node before a start |
| Disk | ~250 GB for the weights (once, or a share all four mount), ~36 GB for the image on every node |
| Software on the nodes | Docker with the NVIDIA runtime, `ibv_devices` (rdma-core), passwordless sudo for the page-cache drop |

## What is in the image

Everything is under `/opt/keys-glm53`; the entry point is `/opt/keys-glm53/serve`.

| Part | Version | Role |
|---|---|---|
| Inference engine | 1.0.7 with our `glm_moe_dsa` CUDA tensor-parallel port | Forward pass, drafting, sampling, prompt reuse; one process a Spark |
| mlx-serve | `main` of 2026-10 with a bridge that loads the engine as a library | HTTP server on rank 0: chat template, tool calls, streaming |
| DSpark ft2 drafter | `draft/dspark-ft2` | Speculative drafts (confidence policy, 0.3) |
| Kernel packs and tile tables | `aot/`, `ref/` | Precompiled decode kernels for 32K context and 4 streams |
| Wide kernel pack | `aot-wide/`, `ref-wide/` | Kernels for the `64k`, `250k` and `1m` presets (built with engine 1.0.4's generator, plus 24 variants for two streams with the cache split) |
| Licenses | `licenses/` | The engine's and mlx-serve's licenses and their third-party notices |

## What this build changes

| Change | Effect |
|---|---|
| Engine merged onto upstream engine 1.0.5 | Upstream's 1.0.5 kernels and fixes under the GLM-5.3 port |
| Prompt reuse keeps a point every 4,096 tokens (`TF_GLM53_REUSE_STRIDE`) | A long prompt with a changed ending resumes from the last shared point: 23 s to 3.3 s on a 25K prompt |
| Engine merged onto upstream engine 1.0.7 (tag `2026-10-11`) | Same replies as `2026-10-10-r4`: in the `1m` preset the prose and code benchmark texts are identical and the speed is the same (35.2 and 44.6 tok/s against 35.0 and 44.7). All checks pass (`evidence/2026-10-11/`). The default and `64k` shapes were not re-measured on this tag with these weights; the tables above are from `2026-10-10-r4` |
| 6-bit output head and a lighter health check (tag `2026-10-12`) | See the section above: code 44.4 to 46.0 tok/s in `PRESET=1m`, KL +0.0001 |
| Two streams with the cache split across the Sparks (tag `2026-10-12-r2`, `PRESET=250k`) | 250,000 tokens × 2 streams; replies identical to one request at a time; prose 34.7 tok/s alone, 46.4 together |
| A client that leaves cancels its reply at once (tag `2026-10-10-r4`) | With one stream a dropped reply used to run to its end and hold the next request (40 s for a 1,400-token reply); now 0.2 to 0.7 s |
| Thinking defaults to High effort | GLM-5.3's template runs at Max effort when no effort is named, which spends thousands of tokens on reasoning. High keeps the reasoning short |
| `enable_thinking: false` is served as Low effort | GLM-5.3's template has no thinking-off form; forcing one made the model rewrite its answer several times |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `NODES` | required | Four fabric addresses, rank 0 first |
| `MODEL` | required | Checkpoint directory, the same path on all four |
| `DRAFTER` | `dspark` | `dspark`: the drafter in the image. `mtp`: GLM-5.3's own MTP layer, no extra model. `dflash`: a DFlash2 directory you supply with `DRAFT=` |
| `PRESET` | unset | `64k`: 65,536 tokens × 2 streams. `250k`: 250,000 tokens × 2 streams. `1m`: 1,000,000 tokens × 1 stream |
| `HEAD_Q` | unset | Unset: the 6-bit output head when `$MODEL/lm_head-6bit/lm_head-6bit.safetensors` exists, else the bf16 head. `off`: the bf16 head |
| `CONTEXT` / `PARALLEL` | the preset's (`32768` / `4`) | Context window and concurrent streams |
| `PORT` | `18992` | HTTP port on rank 0 |
| `RAILS` | `2` | RoCE rails to use; `1` if the second rail is not addressed alike on all four |
| `TF_GLM53_*` | the image's | Any engine setting you export reaches every rank |

Per request: `reasoning_effort` (`low`, `high`, `max`) chooses how long the model reasons.

## Limits

- Four shapes were measured: 32K × 4, 64K × 2, 250K × 2 and 1M × 1. Another context or stream count may need kernels
  the image does not hold. In the default and `64k` shapes each stream holds its own cache on every rank; `250k` and
  `1m` split the cache across the ranks. With the cache split the engine serves at most two streams.
- `PRESET=1m` serves one request at a time; a second request waits. Prompts are read about 2.5 times slower than in the default shape, in `250k` too.
- Exact lookups in a long prompt are less reliable at the default High effort, in every shape. On a 1,400-entry list
  (23K tokens), 82 lookups: High got 73 right at 1M and 76 at 64K, usually returning a neighbouring entry; Max
  (`"reasoning_effort": "max"`) got 81 in both. Use Max when an answer depends on one exact line far back in the
  prompt (`evidence/2026-10-10/lookup-by-effort.log`). Changing the effort re-reads the whole prompt.
- The presets were measured with the DSpark drafter only.
- `DRAFTER=dflash` and `DRAFTER=mtp` ran on this engine before it was packaged, not from this image.
  [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) is CC BY-NC-ND 4.0: you download it yourself.
- Prompt reuse keeps the last shared point only. A prompt that shares less than 4,096 tokens with the one before it is
  read from the start.
- If a rank dies in the middle of a request, the server waits without a timeout. `./one-shot.sh down` and `up` recover.
- The server reports the model as `model` and has no authentication.

## Related

- Credits and licenses: [NOTICE.md](NOTICE.md).
