#!/bin/bash
# Module environment for the ALCF-4 3D ViT benchmark on Aurora.
# Source this from the jobscript (and on a login node to inspect the stack).
#
# This replaces the venv that mds_launch.sh used to activate
# (/lus/flare/projects/Aurora_deployment/eku/venv/vit). That venv was built
# against /opt/aurora/24.347.0, which no longer exists on the machine --
# `import torch` inside it fails. The frameworks module supplies everything
# the benchmark needs; no venv is required.
#
#   frameworks/2026.1.0 provides:
#     python  3.12.12
#     torch   2.13.0a0+gitcf30153   (Intel-patched, XPU backend built in)
#     deepspeed 0.19.3
#     transformers 5.14.1, einops 0.8.2, mpi4py 4.1.2
#
# What is deliberately NOT here, and why it is fine:
#   intel_extension_for_pytorch  -- upstreamed into torch.xpu; the standalone
#                                   package is gone. See the guard in
#                                   megatron/model/__init__.py.
#   oneccl_bindings_for_pytorch  -- replaced by torch's native XCCL backend.
#                                   DeepSpeed detects its absence and selects
#                                   'xccl' (deepspeed/accelerator/xpu_accelerator.py).
#   apex                         -- only imported on the CUDA path.
#   wandb                        -- runs set WANDB_MODE=disabled.

module restore >/dev/null 2>&1
module load frameworks/2026.1.0

# Megatron-DeepSpeed is run from a source checkout, not installed.
# ViT_ROOT is the submodule root; callers set it before sourcing, or we
# fall back to the submodule's usual location relative to this file.
: "${ViT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/Megatron-DeepSpeed_ViT" 2>/dev/null && pwd)}"
export PYTHONPATH="${ViT_ROOT}:${PYTHONPATH}"
