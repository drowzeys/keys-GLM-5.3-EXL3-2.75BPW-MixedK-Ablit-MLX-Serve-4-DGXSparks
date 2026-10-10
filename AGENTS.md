# Instructions for coding agents

You are setting up full GLM-5.3 (EXL3 2.75 bpw, abliterated) on **four** NVIDIA DGX Sparks with the prebuilt image.
The image's defaults are the measured settings: do not add tuning flags. Work through the steps in order, check each
result, and stop and report to your user when a check fails.

## Hard rules

- Four Sparks, one rank each, on a RoCE fabric (ConnectX-7). Fewer nodes do not fit the model.
- GB10 memory is unified: a node that runs out swaps until it hangs. Never start on a node with other GPU jobs or
  less than 100 GB available. `./one-shot.sh check` reports both.
- Do not reboot a node to free memory: stop the other jobs and drop the page cache (`sudo sysctl vm.drop_caches=1`).
- Keep `CONTEXT=32768` with `PARALLEL=4`. That is the only combination measured; each stream holds its own cache on
  every rank, so larger combinations can exhaust memory.
- Run the commands from a machine with passwordless ssh to the four Sparks. It needs no GPU.
- The endpoint has no authentication: keep it on a private network.

## Steps

1. Ask the user for the four fabric addresses, rank 0 first (`NODES`), and a checkpoint path that exists on all four
   (`MODEL`).
2. Weights (once, about 250 GB): `hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated --local-dir $MODEL`
   (gated; approval is automatic after the form).
3. Image, on every node: `docker pull ghcr.io/drowzeys/keys-glm53-exl3-275-ablit-tf-engine-4-dgxsparks:2026-10-10`.
4. Node settings, on every node after every reboot: `sudo node/gb10-node-settings.sh`.
5. `NODES="..." MODEL=... ./one-shot.sh check` must end with `check: OK`.
6. `./one-shot.sh up` (about 6 minutes). It prints `SERVING` with the address, or the failing rank's log lines.
7. `./one-shot.sh bench`: expect about 38 tok/s prose and 56 tok/s code. More than 15 % below that means a node is
   slow: rerun `check`, look for other processes, swap in use or heat.
8. Report the endpoint to the user: `http://<rank 0>:18992/v1`, model name `model`.
