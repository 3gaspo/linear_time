"""Seeded user partitions crossed with TIME's chronological boundaries."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class UserSplit:
    seen: tuple[str, ...]
    unseen: tuple[str, ...]
    seed: int


def split_users(user_ids: Sequence[str], train_fraction: float, seed: int) -> UserSplit:
    users = np.asarray(sorted(set(user_ids)), dtype=str)
    if not len(users) or not 0 < train_fraction <= 1:
        raise ValueError("User splitting needs users and train_fraction in (0, 1]")
    if train_fraction == 1:
        return UserSplit(tuple(users.tolist()), (), int(seed))
    if len(users) < 2:
        # There is no valid held-out-user partition for a singleton dataset.
        return UserSplit(tuple(users.tolist()), (), int(seed))
    order = np.random.default_rng(seed).permutation(len(users))
    count = min(len(users) - 1, max(1, int(len(users) * train_fraction)))
    return UserSplit(
        tuple(sorted(users[order[:count]].tolist())),
        tuple(sorted(users[order[count:]].tolist())),
        int(seed),
    )


def time_intervals(length: int, val_length: int, test_length: int) -> dict[str, tuple[int, int]]:
    """Labels stay inside their interval; contexts may use earlier history."""
    train_end = length - val_length - test_length
    test_start = length - test_length
    if train_end <= 0 or val_length < 0 or test_length <= 0:
        raise ValueError("TIME split lengths must leave a nonempty training prefix")
    return {"train": (0, train_end), "valid": (train_end, test_start), "test": (test_start, length)}
