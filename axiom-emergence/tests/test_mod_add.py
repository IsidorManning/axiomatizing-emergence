import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tasks.mod_add import get_dataloaders

def collect_pairs(loader):
    return set(loader.dataset.pairs)


def test_splits_unique():
    T = 20
    train, val, test, probe = get_dataloaders(T, batch_size=5, seed=0, m=13)
    assert len(train.dataset) == T
    assert len(val.dataset) == T
    assert len(test.dataset) == T
    assert len(probe) == 13 * 13 - 3 * T
    train_pairs = collect_pairs(train)
    val_pairs = collect_pairs(val)
    test_pairs = collect_pairs(test)
    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)


def test_label_noise():
    T = 30
    clean, _, _, _ = get_dataloaders(T, batch_size=10, seed=1, m=17, label_noise=0.0)
    noisy, _, _, _ = get_dataloaders(T, batch_size=10, seed=1, m=17, label_noise=1.0)
    mismatch = False
    for i in range(T):
        x, y = clean.dataset.pairs[i]
        true_label = (x + y) % 17
        _, label = noisy.dataset[i]
        if label.item() != true_label:
            mismatch = True
            break
    assert mismatch


def test_prior_shift():
    T = 40
    train, val, _, _ = get_dataloaders(T, batch_size=10, seed=2, m=11, prior_shift={0: 5.0})
    train_labels = [int(train.dataset[i][1]) for i in range(T)]
    val_labels = [int(val.dataset[i][1]) for i in range(T)]
    assert train_labels.count(0) > val_labels.count(0)
