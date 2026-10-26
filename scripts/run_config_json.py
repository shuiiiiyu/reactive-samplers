"""Run a fully resolved main-experiment JSON config inside a Slurm allocation."""

import json
import sys
from pathlib import Path

from reactive_samplers.cli import run_config


if __name__ == "__main__":
    run_config(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
