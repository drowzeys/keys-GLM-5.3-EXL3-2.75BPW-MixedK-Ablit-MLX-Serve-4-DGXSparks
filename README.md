# keys-GLM 5.3 EXL3 2.75BPW MixedK Ablit · MLX-Serve + TF Engine · 4 DGX Sparks

# Built on [TensorFold 1.0.5](https://github.com/ashhart/TensorFold) by Ash Hart and [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu

The full **GLM-5.3** (753B parameters, EXL3 2.75 bpw mixed-K, abliterated) served across **four NVIDIA DGX Sparks**
(GB10, one rank a Spark over ConnectX-7 RoCE, tensor parallel 4). mlx-serve is the HTTP server on rank 0; TensorFold's
engine does the forward pass, drafting and sampling on all four. One prebuilt image, drafter included.

| | |
|---|---|
| Image | `ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-tf-engine-4-dgxsparks:2026-10-10` (= `latest`) |
| Weights | [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated) (gated, automatic approval) |
| Drafter (in the image) | [drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2), GLM-5.3 License |
| Endpoint | `http://<rank 0>:18992/v1`, OpenAI- and Anthropic-compatible, model name `model`, no authentication |
| Measured configuration | 32K context, 4 streams |

## Performance

Four DGX Sparks, the published image started with `one-shot.sh`, default settings. GLM-5.3 is a reasoning model, so
every number is with thinking on (the server's default, High effort).

| Measurement | Result |
|---|---|
| Prose, one stream (`./one-shot.sh bench`: greedy, 512 tokens) | **38.1 tok/s** |
| Code, one stream (same) | **56.2 tok/s** |
| 25K-token prompt, first time (prefill + answer) | 23.2 s |
| The same 25K prompt with a changed last line (prompt reuse) | **3.3 s** |
| Load time, `up` to serving | 350 s |
| Memory left on each node while serving | 9-10 GB |

Speeds vary by about 9 % from one load to the next on this cluster.

## Checks the image passed

Run with `HOST=<rank 0> python3 bench/checks.py`; logs in `evidence/2026-10-10/`. The last three rows ran on the same
binaries just before they were packaged (`long-replies.log`); the rest ran on the image.

| Check | Result |
|---|---|
| Short answers; `max_tokens=1` | Stops on its own; the limit is honoured |
| Client disconnects 0.05 s, 0.5 s and 3 s into a stream | The next request is answered in under 1 s |
| A title request beside a dropped stream | Both answered |
| 2 dropped streams beside 2 surviving, then 4 at once | All finish on their own |
| Sampled replies (T = 0.8, two seeds) | Clean stop at 440 and 400 tokens |
| Tool call | Correct function and arguments |
| 7 short prompts at once on 4 streams | All finish on their own |
| Code question | Stops at 1,083 tokens |
| Long essay with a 6,000-token limit | Reached the limit while still writing new text (no repetition) |

## Quick start

On a machine with passwordless ssh to the four Sparks:

```sh
git clone https://github.com/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-TF-Engine-4-DGXSparks && cd keys-GLM-5.3-EXL3-2.75BPW-MixedK-Ablit-MLX-Serve-TF-Engine-4-DGXSparks
export NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"      # fabric addresses, rank 0 first
export MODEL=/models/keys-GLM-5.3-EXL3-2.75BPW-Abliterated  # the same path on all four

hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated --local-dir $MODEL    # once, ~250 GB, on every node or a share
for n in $NODES; do ssh $n docker pull ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-tf-engine-4-dgxsparks:2026-10-10; done
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
| TensorFold engine | 1.0.5 with our `glm_moe_dsa` CUDA tensor-parallel port | Forward pass, drafting, sampling, prompt reuse; one process a Spark |
| mlx-serve | `main` of 2026-10 with a bridge that loads the engine as a library | HTTP server on rank 0: chat template, tool calls, streaming |
| DSpark ft2 drafter | `draft/dspark-ft2` | Speculative drafts (confidence policy, 0.3) |
| Kernel packs and tile tables | `aot/`, `ref/` | Precompiled decode kernels for 32K context and 4 streams |
| Licenses | `licenses/` | TensorFold, mlx-serve and their third-party notices |

## What this build changes

| Change | Effect |
|---|---|
| Engine merged onto TensorFold 1.0.5 | Upstream's 1.0.5 kernels and fixes under the GLM-5.3 port |
| Prompt reuse keeps a point every 4,096 tokens (`TF_GLM53_REUSE_STRIDE`) | A long prompt with a changed ending resumes from the last shared point: 23 s to 3.3 s on a 25K prompt |
| Thinking defaults to High effort | GLM-5.3's template runs at Max effort when no effort is named, which spends thousands of tokens on reasoning. High keeps the reasoning short |
| `enable_thinking: false` is served as Low effort | GLM-5.3's template has no thinking-off form; forcing one made the model rewrite its answer several times |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `NODES` | required | Four fabric addresses, rank 0 first |
| `MODEL` | required | Checkpoint directory, the same path on all four |
| `DRAFTER` | `dspark` | `dspark`: the drafter in the image. `mtp`: GLM-5.3's own MTP layer, no extra model. `dflash`: a DFlash2 directory you supply with `DRAFT=` |
| `CONTEXT` / `PARALLEL` | `32768` / `4` | Context window and concurrent streams |
| `PORT` | `18992` | HTTP port on rank 0 |
| `RAILS` | `2` | RoCE rails to use; `1` if the second rail is not addressed alike on all four |
| `TF_GLM53_*` | the image's | Any engine setting you export reaches every rank |

Per request: `reasoning_effort` (`low`, `high`, `max`) chooses how long the model reasons.

## Limits

- Only 32K context with 4 streams was measured. The kernel pack in the image was built for that configuration; another
  context or stream count may need kernels the image does not hold, and each stream holds its own cache on every rank.
- `DRAFTER=dflash` and `DRAFTER=mtp` ran on this engine before it was packaged, not from this image.
  [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) is CC BY-NC-ND 4.0: you download it yourself.
- Prompt reuse keeps the last shared point only. A prompt that shares less than 4,096 tokens with the one before it is
  read from the start.
- If a rank dies in the middle of a request, the server waits without a timeout. `./one-shot.sh down` and `up` recover.
- The server reports the model as `model` and has no authentication.

## Related

- [keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark):
  the same weights on TensorFold's Python engine, up to 1M context.
- Credits and licenses: [NOTICE.md](NOTICE.md).
