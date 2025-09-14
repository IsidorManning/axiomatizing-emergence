import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any

class ModAddDataset(Dataset):
    """Dataset for modular addition.

    Each item is a pair ``(x, y)`` with integers ``0 <= x, y < m`` and
    label ``(x + y) % m``. Optional ``label_noise`` corrupts labels with
    probability ``label_noise`` by replacing them with a random label.
    """

    def __init__(self, pairs: List[Tuple[int, int]], m: int, *, label_noise: float = 0.0, seed: int | None = None) -> None:
        self.pairs = pairs
        self.m = m
        self.label_noise = label_noise
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x, y = self.pairs[idx]
        label = (x + y) % self.m
        if self.label_noise > 0 and self.rng.random() < self.label_noise:
            label = int(self.rng.integers(self.m))
        inputs = torch.tensor([x, y], dtype=torch.long)
        target = torch.tensor(label, dtype=torch.long)
        return inputs, target


def _sample_split(
    pairs: List[Tuple[int, int]],
    labels: np.ndarray,
    size: int,
    rng: np.random.Generator,
    *,
    weights: np.ndarray | None = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], np.ndarray]:
    """Sample ``size`` unique pairs from ``pairs`` using ``weights``.

    Returns the selected pairs, remaining pairs and remaining labels.
    """

    idx = np.arange(len(pairs))
    chosen = rng.choice(idx, size=size, replace=False, p=weights)
    remaining = np.setdiff1d(idx, chosen, assume_unique=True)
    selected_pairs = [pairs[i] for i in chosen]
    remaining_pairs = [pairs[i] for i in remaining]
    remaining_labels = labels[remaining]
    return selected_pairs, remaining_pairs, remaining_labels


def get_dataloaders(
    T: int,
    batch_size: int,
    seed: int,
    *,
    m: int = 97,
    **jitter: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader, ModAddDataset]:
    """Create dataloaders for the modular addition task.

    Parameters
    ----------
    T: int
        Number of unique samples in each of the train/val/test splits.
    batch_size: int
        Batch size for the dataloaders.
    seed: int
        Random seed controlling the split.
    m: int, optional
        Modulus for addition. Defaults to ``97``.
    jitter: dict, optional
        Optional jitter parameters. Supported keys:

        - ``label_noise``: probability of label corruption in the training set.
        - ``prior_shift``: dict mapping labels to relative sampling weights
          for the training set.

    Returns
    -------
    train_loader, val_loader, test_loader, probe_dataset
        Three dataloaders and a probe dataset containing the remaining pairs.
    """

    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(m) for j in range(m)]
    labels = np.array([(x + y) % m for x, y in all_pairs])

    if 3 * T > len(all_pairs):
        raise ValueError("T too large for modulus m")

    # Prior shift for training split
    prior_shift = jitter.get("prior_shift")
    weights = None
    if prior_shift:
        weights = np.array([prior_shift.get(lbl, 1.0) for lbl in labels], dtype=float)
        weights = weights / weights.sum()

    train_pairs, remaining_pairs, remaining_labels = _sample_split(
        all_pairs, labels, T, rng, weights=weights
    )
    val_pairs, remaining_pairs, remaining_labels = _sample_split(
        remaining_pairs, remaining_labels, T, rng
    )
    test_pairs, remaining_pairs, _ = _sample_split(
        remaining_pairs, remaining_labels, T, rng
    )

    probe_pairs = remaining_pairs

    label_noise = float(jitter.get("label_noise", 0.0))
    train_ds = ModAddDataset(train_pairs, m, label_noise=label_noise, seed=rng.integers(1 << 32))
    val_ds = ModAddDataset(val_pairs, m, seed=rng.integers(1 << 32))
    test_ds = ModAddDataset(test_pairs, m, seed=rng.integers(1 << 32))
    probe_ds = ModAddDataset(probe_pairs, m, seed=rng.integers(1 << 32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, probe_ds
