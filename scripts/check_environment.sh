#!/usr/bin/env bash
# Install only the small framework dependencies. Reuse system torch for scheduler testing.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p sampler_runs/setup/logs
exec > sampler_runs/setup/logs/environment-check.log 2>&1
status=0
uv sync || status=$?
if test "$status" -eq 0; then
  uv run python -m compileall -q reactive_samplers scripts tests || status=$?
  uv run reactive-samplers --help || status=$?
  uv run python scripts/run_main.py --cfg job --resolve || status=$?
  # Query the unactivated host interpreter: Ubuntu uses dist-packages, not site-packages.
  system_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  PYTHONPATH="$PWD:$system_packages" uv run --no-project --with diffusers==0.35.1 --with hydra-core==1.3.2 \
    python -m unittest discover -s tests -v || status=$?
fi
printf '%s\n' "$status" > sampler_runs/setup/logs/environment-check.exit-code
exit "$status"
