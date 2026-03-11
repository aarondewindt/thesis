# Toy Multi-Phase RLlib Project (Conveyor–Portal)

This repo scaffolds **Milestone A** (dev environment + reproducible project skeleton) for the **Conveyor–Portal sequential-policy toy project** from the project chat PDF.

## What this is (Milestone A)
- A clean Python project layout (src/ style) with placeholders for env/training/experiments/tests.
- A VS Code **devcontainer** that is ROCm-ready for AMD GPUs (passes `/dev/kfd` + `/dev/dri` into the container).
- Basic scripts for formatting/linting/testing.
- **No environment or RL code yet**.

## Prerequisites on the host (Ubuntu 24.04)
1. Install AMD ROCm drivers on the host and confirm the GPU is visible.
2. Docker installed and working.
3. VS Code + “Dev Containers” extension.

Notes:
- ROCm containers typically need access to **/dev/kfd** and **/dev/dri**. The host user running Docker often needs to be in the **render** group so VS Code can run the container without root.
- ROCm containers are documented by AMD and ROCm dev images are available on Docker Hub. citeturn0search15turn0search0

## Opening in VS Code
- Open this folder in VS Code
- Command Palette → **Dev Containers: Reopen in Container**

## Quick checks (after build)
Inside the container:
```bash
python -V
python -c "import ray; import gymnasium; print('ok')"
pytest -q
```

### GPU sanity checks (inside the container)
If ROCm tools are present, these help:
```bash
ls -l /dev/kfd /dev/dri || true
rocminfo | head -n 40 || true
```

TensorFlow GPU detection will be validated once we pin a ROCm-compatible TF wheel in Milestone B/C.
AMD’s ROCm TensorFlow docs include additional notes (e.g., `tf-keras` for Keras 2 compatibility). citeturn0search13

## Layout
- `src/toy_mp/` — package root (envs, rllib wiring, experiments)
- `tests/` — unit tests and smoke tests
- `.devcontainer/` — devcontainer config + Dockerfile
- `scripts/` — helper scripts

## Next: Milestone A tasks
See `docs/milestone_a_tasks.md`.


asdas
