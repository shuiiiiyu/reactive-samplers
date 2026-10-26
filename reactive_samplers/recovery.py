"""Step-local recovery and the shared human/Agent failure ledger.

Policy code is trusted Python, not a security sandbox. SIGKILL, native crashes and
wedged CUDA contexts cannot be recovered by Python exception handling.
"""
import copy
import contextlib
import dataclasses
import datetime
import json
import math
from pathlib import Path
import random
import signal
import sys
import threading
import time
import traceback
import uuid

from .report import write_json


POLICY_ERRORS = (Exception, SystemExit)


@contextlib.contextmanager
def policy_deadline(seconds=30):
    """Bound Python policy code on the Linux sampler's main thread."""
    enabled = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not enabled:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    remaining, interval = signal.getitimer(signal.ITIMER_REAL)
    started = time.perf_counter()
    def expired(signum, frame):
        raise TimeoutError(f"Policy execution exceeded {seconds:g} seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, remaining) if remaining else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if remaining:
            signal.setitimer(signal.ITIMER_REAL, max(0, remaining - (time.perf_counter() - started)), interval)


def clone_state(value):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: clone_state(getattr(value, f.name))
                                            for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    state = {"python": random.getstate()}
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_initialized():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    torch = sys.modules.get("torch")
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def require_finite(value, name):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f"{name} is non-finite")


def fatal_error(exc):
    # Retrying a damaged CUDA context can corrupt results or hang the process.
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or any(text in message for text in (
        "out of memory", "cuda error", "device-side assert", "illegal memory access",
        "unspecified launch failure", "cuda context",
    ))


class FailureLog:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / "failures.jsonl"
        self.records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()] if path.exists() else []
        self.detected_at = {}

    def record(self, exc, *, stage, ctx=None, version=None, outcome="rejected",
               restored=False, target=None, memory="unchanged", decision=None):
        record = {
            "id": uuid.uuid4().hex, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "sample_id": getattr(ctx, "sample_id", ""), "seed": getattr(ctx, "seed", None),
            "step": getattr(ctx, "step", None), "nfe": getattr(ctx, "nfe", None),
            "sigma": getattr(ctx, "sigma", None), "timestep": getattr(ctx, "timestep", None),
            "stage": stage, "version": version, "exception": type(exc).__name__, "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-12000:],
            "outcome": outcome, "sampling_state_restored": restored, "target_version": target,
            "policy_memory": memory, "decision": decision, "continued": False,
            "prior_committed_steps_retained": True, "extra_nfe": 0,
            "recovery_seconds": None,
        }
        self.detected_at[record["id"]] = time.perf_counter()
        self.records.append(record)
        self._write()
        return record

    def recovered(self, records, version):
        for record in records:
            record.update(continued=True, effective_version=version)
            record["recovery_seconds"] = time.perf_counter() - self.detected_at[record["id"]]
        if records:
            self._write()

    def _write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        # One writer: the paused sampler owns this ledger, including diagnostics.
        path = self.root / "failures.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                     for row in self.records), encoding="utf-8")
        temporary.replace(path)
        lines = ["# Failure and recovery report", "",
                 "Restoration concerns the current uncommitted step; earlier committed steps are retained.", ""]
        for row in self.records:
            lines += [f"## {row['sample_id'] or 'run'} — {row['stage']} ({row['id']})", "",
                      f"- Step / NFE / sigma: {row['step']} / {row['nfe']} / {row['sigma']}",
                      f"- Version / recovery target: {row['version']} / {row['target_version']}",
                      f"- Outcome: {row['outcome']}; state restored: {row['sampling_state_restored']}; "
                      f"memory: {row['policy_memory']}; continued: {row['continued']}",
                      f"- Extra NFE / seconds from failure detection to successful step: {row['extra_nfe']} / "
                      f"{row.get('recovery_seconds')}", "", "```text", row["traceback"], "```", ""]
        report = self.root / "failure_report.md"
        temporary = report.with_suffix(".md.tmp")
        temporary.write_text("\n".join(lines), encoding="utf-8")
        temporary.replace(report)


class PolicyRecovery:
    """Versions earn fallback eligibility after two successful committed updates."""
    def __init__(self, programs):
        self.programs = programs
        if not hasattr(programs, "recovery_versions"):
            programs.recovery_versions = [programs.version]
            programs.recovery_successes = {}
            programs.recovery_bad = set()

    def success(self):
        programs = self.programs
        count = programs.recovery_successes.get(programs.version, 0) + 1
        programs.recovery_successes[programs.version] = count
        if count >= 2 and programs.version not in programs.recovery_versions:
            programs.recovery_versions.append(programs.version)

    def fallback(self):
        programs = self.programs
        self.restore_failures = []
        programs.recovery_bad.add(programs.version)
        for version in reversed(programs.recovery_versions):
            if version not in programs.recovery_bad:
                try:
                    programs.restore(version)
                    return False
                except POLICY_ERRORS as exc:
                    if fatal_error(exc):
                        raise
                    programs.recovery_bad.add(version)
                    self.restore_failures.append((version, exc))
        programs.use_default()
        return True


def save_crash(adapter, ctx, recorder, programs, error):
    """Save a trusted diagnostic snapshot, not a claim of automatic job resumption."""
    info = {"status": "aborted", "error": str(error), "version": programs.version,
            "sample_id": getattr(ctx, "sample_id", ""), "state_saved": False}
    if ctx is not None and hasattr(adapter, "save_failure_state"):
        try:
            path = recorder.root / "failure-state.pt"
            adapter.save_failure_state(ctx, path)
            info.update(state_saved=True, state=str(path))
        except POLICY_ERRORS as exc:
            info["snapshot_error"] = str(exc)
    write_json(recorder.root / "failure.json", info)
