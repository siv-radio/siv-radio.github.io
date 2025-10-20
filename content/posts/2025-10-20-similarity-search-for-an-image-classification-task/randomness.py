"""
Tools to make results reproducible.

References:
1. "Reproducibility", PyTorch v2.7, updated on 2024.11.26.
   https://docs.pytorch.org/docs/stable/notes/randomness.html
2. "2.1.4. Results Reproducibility",
   cuBLAS v12.9, "1. Introduction", updated on 2025.05.03.
   https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
3. "Random seeds and reproducible results in PyTorch",
   by Vandana Rajan, 2021.05.11.
   https://vandurajan91.medium.com/random-seeds-and-reproducible-results-in-pytorch-211620301eba
4. "How to Set Random Seeds in PyTorch and TensorFlow",
   by Hey Amit, 2024.12.06. Note: probably, AI-generated, but useful.
   https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
"""

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["make_torch_generator", "seed_worker", "set_determinism"]


def set_determinism(*, seed: int, use_deterministic_algorithms: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(
    *,
    seed: Optional[int] = None,
    device: Optional[str] = None
) -> torch._C.Generator:
    if device is not None:
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator


# An example:
#
# generator = make_torch_generator(seed=random_seed)
#
# DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     num_workers=num_workers,
#     worker_init_fn=seed_worker,
#     generator=generator,
# )
