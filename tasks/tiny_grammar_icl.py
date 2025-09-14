"""Tiny grammar in-context learning dataset.

This module generates sequences for a toy balanced-parentheses grammar
task.  Each sequence consists of `k` demonstration examples followed by
a query example whose label must be predicted by a transformer.

The dataset produces token and label lists.  Tokens are integers
representing characters from a small vocabulary::

    0: '(' 
    1: ')' 
    2: '0' (label for unbalanced)
    3: '1' (label for balanced)
    4: '?' (query marker)
    5: ' ' (separator)

Labels are -100 for tokens that are part of the context (demo
examples).  The only supervised position is the final ``?`` token whose
label is either ``2`` or ``3`` (i.e. ``'0'`` or ``'1'``).

Jitter hooks: ``set_jitter_hooks`` allows the caller to inject
functions that modify the token or label sequences, mirroring the hooks
used in the modular addition task.  By default these hooks are identity
functions.

Example
-------

>>> ds = TinyGrammarICL(L=6, k=2, seed=0)
>>> tokens, labels = ds.make_example()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import random

# Vocabulary used by the task.
VOCAB = {"(": 0, ")": 1, "0": 2, "1": 3, "?": 4, " ": 5}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def generate_balanced_parentheses(max_len: int) -> List[str]:
    """Generate all balanced-parentheses strings up to ``max_len``.

    ``max_len`` is the maximum length (must be even).  The empty string
    is not included in the result.
    """
    results: set[str] = set()

    def backtrack(prefix: str, open_count: int, close_count: int) -> None:
        if len(prefix) > max_len:
            return
        if open_count == 0 and close_count == 0 and prefix:
            results.add(prefix)
        if len(prefix) == max_len:
            return
        if open_count < max_len // 2:
            backtrack(prefix + "(", open_count + 1, close_count)
        if close_count < open_count:
            backtrack(prefix + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return sorted(results, key=len)


def _is_balanced(s: str) -> bool:
    """Return ``True`` if ``s`` is a balanced-parentheses string."""
    count = 0
    for ch in s:
        if ch == "(":
            count += 1
        else:
            count -= 1
        if count < 0:
            return False
    return count == 0


@dataclass
class TinyGrammarICL:
    """Dataset generator for the tiny balanced-parentheses grammar.

    Parameters
    ----------
    L:
        Maximum string length (even).
    k:
        Number of demonstration examples to include.
    seed:
        Optional random seed.
    """

    L: int = 6
    k: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self._token_hook: Callable[[List[int]], List[int]] = lambda x: x
        self._label_hook: Callable[[List[int]], List[int]] = lambda x: x

    # ------------------------------------------------------------------ #
    # Jitter hooks
    # ------------------------------------------------------------------ #
    def set_jitter_hooks(
        self,
        token_hook: Callable[[List[int]], List[int]] | None = None,
        label_hook: Callable[[List[int]], List[int]] | None = None,
    ) -> None:
        """Register jitter hooks for tokens and labels.

        Hooks receive and must return a list of integers.
        """
        if token_hook is not None:
            self._token_hook = token_hook
        if label_hook is not None:
            self._label_hook = label_hook

    # ------------------------------------------------------------------ #
    # String generation utilities
    # ------------------------------------------------------------------ #
    def _random_balanced(self, n_pairs: int) -> str:
        """Return a random balanced string with ``n_pairs`` pairs."""
        s: List[str] = []
        balance = 0
        opens_remaining = n_pairs
        for _ in range(2 * n_pairs):
            if opens_remaining == 0:
                s.append(")")
                balance -= 1
            elif balance == 0:
                s.append("(")
                opens_remaining -= 1
                balance += 1
            elif self.rng.random() < 0.5:
                s.append("(")
                opens_remaining -= 1
                balance += 1
            else:
                s.append(")")
                balance -= 1
        return "".join(s)

    def _random_unbalanced(self, n_pairs: int) -> str:
        """Return a random *unbalanced* string with ``n_pairs`` pairs."""
        s = list(self._random_balanced(n_pairs))
        # Flip a random position to break balance.
        idx = self.rng.randrange(len(s))
        s[idx] = "(" if s[idx] == ")" else ")"
        return "".join(s)

    def _sample_string(self) -> Tuple[str, int]:
        """Sample a random string and its label (1=balanced,0=unbalanced)."""
        n_pairs = self.rng.randint(1, self.L // 2)
        if self.rng.random() < 0.5:
            s = self._random_balanced(n_pairs)
            label = 1
        else:
            s = self._random_unbalanced(n_pairs)
            label = 0
        return s, label

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def make_example(self) -> Tuple[List[int], List[int]]:
        """Create a single ICL example.

        Returns
        -------
        tokens:
            List of input token ids.
        labels:
            List of label ids (-100 for tokens that should be ignored).
        """
        tokens: List[str] = []
        labels: List[int] = []

        # Demonstration examples
        for _ in range(self.k):
            s, lbl = self._sample_string()
            tokens.extend(list(s) + [" ", str(lbl), " "])
            labels.extend([-100] * (len(s) + 3))  # context, ignore

        # Query example
        query, q_label = self._sample_string()
        tokens.extend(list(query) + [" ", "?"])
        labels.extend([-100] * (len(query) + 1))
        labels.append(VOCAB[str(q_label)])

        token_ids = [VOCAB[t] for t in tokens]
        label_ids = labels

        # Apply jitter hooks
        token_ids = self._token_hook(token_ids)
        label_ids = self._label_hook(label_ids)

        return token_ids, label_ids
