import pytest


tasks = pytest.importorskip("tasks")

if not hasattr(tasks, "load_tiny_batch"):
    pytest.skip("tasks.load_tiny_batch missing", allow_module_level=True)


def test_tiny_batch_shapes_and_labels():
    inputs, labels, num_classes = tasks.load_tiny_batch(batch_size=2)
    assert inputs.shape[0] == 2
    assert labels.shape[0] == 2
    assert labels.min().item() >= 0
    assert labels.max().item() < num_classes
