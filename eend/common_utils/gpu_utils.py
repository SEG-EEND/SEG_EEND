#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Copyright 2025 Human Interface Lab (author: C. Moon)
# Licensed under the MIT license.

from safe_gpu import safe_gpu


def use_single_gpu(gpus_qty: int) -> safe_gpu.GPUOwner:
    assert gpus_qty < 2, "Multi-GPU still not available."
    gpu_owner = safe_gpu.GPUOwner(nb_gpus=gpus_qty)
    return gpu_owner
