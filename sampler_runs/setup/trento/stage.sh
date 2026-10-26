#!/usr/bin/env bash
set -euo pipefail
cd /workspace/home/shencanyu/deepresearch/workspace/reactive-samplers
state=sampler_runs/setup/trento
mkdir -p "$state/wheels"
exec > "$state/stage.log" 2>&1
printf 'STAGING\n' > "$state/stage.status"
uv python install 3.11
test -x "$state/resolver-env/bin/python" || uv venv --python 3.11 "$state/resolver-env"
uv pip compile --python "$state/resolver-env/bin/python" --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match tests/trento-requirements.in --output-file tests/trento-requirements.lock
uv pip install --python "$state/resolver-env/bin/python" pip
"$state/resolver-env/bin/python" -m pip wheel --extra-index-url https://download.pytorch.org/whl/cu121 --wheel-dir "$state/wheels" -r tests/trento-requirements.lock
sha256sum "$state"/wheels/* > "$state/wheels.sha256"
printf 'STAGED\n' > "$state/stage.status"
