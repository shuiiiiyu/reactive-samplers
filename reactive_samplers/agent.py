"""A decision pauses sampling; the agent may rerun inspect repeatedly before returning."""
import time
import uuid
import hashlib
from pathlib import Path

from .programs import Programs
from .report import write_json
from .recovery import clone_state, policy_deadline, rng_state, restore_rng
from .context import METHODS


class AgentSession:
    def __init__(self, backend, run_dir, save_context, method="adaptive_sigma"):
        if method not in METHODS:
            raise ValueError(f"Unknown sampling method: {method}")
        self.method = method
        self.instructions = Path(__file__).parent / "prompts" / (
            "grid_skip/sampling_agent.md" if method == "grid_skip" else "sampling_agent.md")
        self.backend = backend
        self.root = Path(run_dir).resolve()
        self.save_context = save_context
        self.count = 0
        self.seconds = 0.0
        self.previous_outcome = None
        self.failure_log = None
        self.failure_cursor = 0
        self.current_folder = None
        self.outcome_folder = None

    def record_outcome(self, status, ctx, version, failure_ids):
        self.previous_outcome = {"decision": self.count, "outcome": status, "step": ctx.step,
                                 "sample_id": ctx.sample_id, "version": version,
                                 "failure_ids": failure_ids}
        if self.current_folder is not None:
            write_json(self.current_folder / "outcome.json", self.previous_outcome)
            self.outcome_folder = self.current_folder

    def mark_reverted(self, version, failure):
        if self.previous_outcome and self.previous_outcome.get("version") == version:
            self.previous_outcome.update(outcome="reverted", reverted_at=failure["step"],
                                         failure_ids=[*self.previous_outcome["failure_ids"], failure["id"]])
            if self.outcome_folder is not None:
                write_json(self.outcome_folder / "outcome.json", self.previous_outcome)

    def check(self, ctx, observation, programs):
        if ctx.method != self.method or programs.method != self.method:
            raise ValueError("Agent, context and policy methods must match")
        self.count += 1
        folder = self.root / "decisions" / f"{self.count:05d}"
        folder.mkdir(parents=True)
        self.current_folder = folder
        state = folder / "state.pt"
        self.save_context(ctx, state)
        request = folder / "request.json"
        diagnostic_workspace = folder / "diagnostic_workspace"
        write_json(request, {
            "protocol_version": 2, "request_id": uuid.uuid4().hex,
            "session_id": hashlib.sha256(str(self.root).encode()).hexdigest()[:16], "decision": self.count,
            "method": self.method, "model_index": ctx.model_index, "candidate_index": ctx.candidate_index,
            "sample_id": ctx.sample_id,
            "step": ctx.step, "nfe": ctx.nfe, "sigma": ctx.sigma,
            "state": str(state), "workspace": str(programs.workspace.resolve()),
            "diagnostic_workspace": str(diagnostic_workspace.resolve()),
            "policy_version": programs.version, "observation": observation,
            "instructions": str(self.instructions.resolve()),
            "inspect_command": ["reactive-samplers", "inspect", str(state), "--workspace", str(programs.workspace.resolve())],
            "feedback": str(self.root / "feedback.json"),
            "previous_answer": self.previous_outcome,
            "failures": self.failure_log.records[self.failure_cursor:][-10:] if self.failure_log else [],
            "failure_report": str(self.root / "failure_report.md") if (self.root / "failure_report.md").is_file() else None,
            "failure_log": str(self.root / "failures.jsonl") if (self.root / "failures.jsonl").is_file() else None,
        })
        if self.failure_log:
            self.failure_cursor = len(self.failure_log.records)
        started = time.perf_counter()
        try:
            def inspect(candidate_workspace):
                candidate = Path(candidate_workspace).resolve()
                if candidate != diagnostic_workspace.resolve():
                    raise ValueError("Diagnostic workspace must be this decision's isolated workspace")
                state = rng_state()
                try:
                    probe = Programs(candidate, folder / "diagnostic_programs", method=self.method)
                    isolated = clone_state(ctx)
                    with policy_deadline():
                        observation = probe.analysis.analyze(isolated, clone_state(ctx.analysis_memory))
                    # Diagnostic observations obey the same serialization contract.
                    import json
                    if not isinstance(observation, dict):
                        raise TypeError("Diagnostic analysis must return a dictionary")
                    json.dumps(observation, allow_nan=False)
                    return observation
                except (Exception, SystemExit) as exc:
                    if self.failure_log:
                        self.failure_log.record(exc, stage="diagnostic", ctx=ctx,
                                                version=programs.version, outcome="diagnostic_error",
                                                restored=True, decision=self.count)
                    raise
                finally:
                    restore_rng(state)

            if hasattr(self.backend, "check"):
                answer = self.backend.check(request, programs.workspace.resolve(), inspect)
            else:
                answer = self.backend(request, programs.workspace.resolve())
            message = answer.get("message", "") if isinstance(answer, dict) else answer
            (folder / "answer.txt").write_text(message or "", encoding="utf-8")
            return answer
        finally:
            elapsed = time.perf_counter() - started
            self.seconds += elapsed
            write_json(folder / "timing.json", {"seconds": elapsed})
