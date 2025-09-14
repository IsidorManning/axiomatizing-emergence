import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from tasks.tiny_grammar_icl import TinyGrammarICL, VOCAB


def test_example_shapes():
    ds = TinyGrammarICL(L=6, k=2, seed=0)
    tokens, labels = ds.make_example()
    assert len(tokens) == len(labels)
    assert labels[-1] in (VOCAB["0"], VOCAB["1"])
    assert all(l == -100 for l in labels[:-1])
    assert all(t in VOCAB.values() for t in tokens)
