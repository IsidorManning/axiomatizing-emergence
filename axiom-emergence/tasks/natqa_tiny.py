import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

# Small SST-2-like subset
_SST2_LITE: Dict[str, List[Tuple[str, int]]] = {
    "train": [
        ("a delightful romantic comedy", 1),
        ("utterly dull and predictable", 0),
        ("an astonishing piece of work", 1),
        ("boring documentary with no surprises", 0),
        ("truly inspiring and heartwarming", 1),
        ("too long and very tedious", 0),
        ("smart bold and funny", 1),
        ("flat characters and a weak story", 0),
    ],
    "val": [
        ("an enjoyable family film", 1),
        ("no point or charm", 0),
        ("rich in suspense and drama", 1),
        ("fails to make sense", 0),
    ],
    "test": [
        ("funniest movie of the year", 1),
        ("a total waste of time", 0),
        ("simply beautiful storytelling", 1),
        ("the plot is terrible", 0),
    ],
    "probe": [
        ("i loved it", 1),
        ("i hated it", 0),
        ("not bad at all", 1),
        ("nothing special", 0),
    ],
}

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return text.lower().split()

def build_vocab(samples: List[Tuple[str, int]]) -> Dict[str, int]:
    """Construct a vocabulary from the training samples."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for text, _ in samples:
        for tok in tokenize(text):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert text into token ids."""
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

def jitter_label_flip(data: List[Tuple[List[int], int]], prob: float, rng: random.Random):
    """Randomly flip labels with probability ``prob``."""
    if prob <= 0:
        return data
    jittered = []
    for tokens, label in data:
        if rng.random() < prob:
            label = 1 - label
        jittered.append((tokens, label))
    return jittered

def jitter_prior_shift(data: List[Tuple[List[int], int]], shift: float, rng: random.Random):
    """Adjust class prior by resampling examples."""
    if shift == 0:
        return data
    pos = [d for d in data if d[1] == 1]
    neg = [d for d in data if d[1] == 0]
    total = len(data)
    base_ratio = len(pos) / total
    desired_ratio = min(max(base_ratio + shift, 0), 1)
    desired_pos = int(round(total * desired_ratio))
    desired_neg = total - desired_pos
    if not pos or not neg:
        return data
    new_pos = [rng.choice(pos) for _ in range(desired_pos)]
    new_neg = [rng.choice(neg) for _ in range(desired_neg)]
    new_data = new_pos + new_neg
    rng.shuffle(new_data)
    return new_data

class SST2LiteDataset(Dataset):
    def __init__(self, split: str, vocab: Dict[str, int], label_flip: float = 0.0,
                 prior_shift: float = 0.0, seed: int = 0):
        if split not in _SST2_LITE:
            raise ValueError(f"Unknown split {split}")
        rng = random.Random(seed)
        data = [(encode(text, vocab), label) for text, label in _SST2_LITE[split]]
        if split == "train" and prior_shift != 0.0:
            data = jitter_prior_shift(data, prior_shift, rng)
        if label_flip != 0.0:
            data = jitter_label_flip(data, label_flip, rng)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return tokens, label

def collate_batch(batch):
    tokens, labels = zip(*batch)
    max_len = max(len(t) for t in tokens)
    padded = [t + [0] * (max_len - len(t)) for t in tokens]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def get_loaders(batch_size: int = 4, label_flip: float = 0.0,
               prior_shift: float = 0.0, seed: int = 0):
    vocab = build_vocab(_SST2_LITE["train"])
    train_ds = SST2LiteDataset("train", vocab, label_flip, prior_shift, seed)
    val_ds = SST2LiteDataset("val", vocab, seed=seed)
    test_ds = SST2LiteDataset("test", vocab, seed=seed)
    probe_ds = SST2LiteDataset("probe", vocab, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    probe_loader = DataLoader(probe_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader, test_loader, probe_loader, vocab
