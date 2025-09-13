# Axiom Emergence

## Quickstart
1. Install dependencies
   ```bash
   make setup
   ```
2. Run tests
   ```bash
   make test
   ```
3. Launch a fast smoke test for scaling
   ```bash
   make run-scaling-smoke
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
