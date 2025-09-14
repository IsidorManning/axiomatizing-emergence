# Axiom Emergence

## Quickstart

To install dependencies, run the tests, and execute a tiny scaling smoke
test, run the following commands from this directory:

```bash
make setup
make test
AE_SMALL=1 make run-scaling-smoke
```

## Experiments
Run any of the following to reproduce placeholder experiments:

| Command | Description | Expected Output |
|--------|-------------|----------------|
| `make run-scaling` | full scaling experiment | `Running scaling experiment...` |
| `make run-scaling-smoke` | quick scaling smoke test | `Running scaling smoke test...` |
| `make run-repr` | representation experiment | `Running representation experiment...` |
| `make run-kernel` | kernel experiment | `Running kernel experiment...` |
| `make plots` | generate plots | `Generating plots (placeholder)...` |

Set the environment variable `AE_SMALL=1` to use a reduced configuration for quicker iteration. The `run-scaling-smoke` target enables this knob automatically.
