"""Deterministic seeding helpers for tiny training runs."""

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible smoke tests."""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
