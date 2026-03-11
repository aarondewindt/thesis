# Milestone A — Dev environment + scaffold

## Goal
Open this repo in a VS Code devcontainer and be able to:
- install deps in a venv
- run tests
- (optionally) see ROCm devices in the container

## Tasks
1. **Host prerequisites**
   - Confirm ROCm drivers are installed on host.
   - Confirm `/dev/kfd` exists and your user has access (typically `render` group).
   - Confirm Docker works.

2. **Devcontainer build**
   - Open in VS Code → “Reopen in Container”.
   - Verify `python -V` and `pip -V`.

3. **Dependency install**
   - `pip install -e ".[dev]"` (editable + dev deps)

4. **Smoke tests**
   - `pytest`
   - `python -c "import ray; import gymnasium; print('imports ok')"`

5. **(Optional) ROCm visibility checks**
   - `ls -l /dev/kfd /dev/dri`
   - `rocminfo | head -n 40` (if present)

6. **Decide TF strategy for Milestone B**
   - Pin a ROCm-compatible TensorFlow wheel version for your ROCm release.
   - Add to dependencies and add a `tf.config.list_physical_devices('GPU')` check.

## Definition of done
- Devcontainer builds.
- `pip install -e ".[dev]"` succeeds.
- `pytest` runs.
- (Optional) `/dev/kfd` + `/dev/dri` visible in container.
