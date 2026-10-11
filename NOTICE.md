# Notice

This repository (scripts, benches, documentation): Copyright 2026 drowzeys, MIT License (`LICENSE`). The image holds
separate works under their own licenses; their license texts are in the image under `/opt/keys-glm53/licenses`.

## In the image

| Work | Author | License | What it is here |
|---|---|---|---|
| TensorFold 1.0.5 (third-party software, redistributed in modified form) | Ash Hart and the TensorFold contributors | Apache 2.0 (code before v0.6.0: MIT) | The inference engine: forward, drafting, sampling, prompt reuse. Modified by us: built with our `glm_moe_dsa` CUDA tensor-parallel port and a C interface (`libtensorfold_glm53.so`) |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | David Dalcu | MIT (third-party parts listed in its `NOTICE`) | The HTTP server on rank 0: OpenAI and Anthropic endpoints, chat template, tool calls. Built with a bridge that loads the engine above |
| [drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2) | drowzeys, from [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) | GLM-5.3 License (in its directory) | The drafter, at `/opt/keys-glm53/draft/dspark-ft2` |
| Base image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-05` | see [its notice](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark/blob/main/NOTICE.md) | as listed there | CUDA, NCCL and cuBLAS libraries (NVIDIA's terms apply to the NVIDIA software) |

The TensorFold project is not affiliated with this repository and does not endorse or test this model.

The GLM-5.3 port builds on work credited in the base image's notice: MiaAI-Lab, Jay Leaton, BertholomusAI (Albert Lee),
cuda-exl3, b12x and ExLlamaV3 (the EXL3 format, Turboderp). Engineering assistance: Anthropic's Claude (Claude Code).

## Models (not in the image or this repository)

- **GLM-5.3**: Z.ai (zai-org), GLM-5.3 License. The quantized, abliterated weights are
  [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated).
- **DFlash2 (optional)**: [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2), CC BY-NC-ND 4.0
  (non-commercial, no derivatives). It is not redistributed here; users download it themselves.
