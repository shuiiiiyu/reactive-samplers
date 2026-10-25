"""A decision pauses sampling; the agent may rerun inspect repeatedly before returning."""
from pathlib import Path
import time

from .report import write_json


class AgentSession:
    def __init__(self, backend, run_dir, save_context):
        self.backend = backend
        self.root = Path(run_dir).resolve()
        self.save_context = save_context
        self.count = 0
        self.seconds = 0.0

    def check(self, ctx, observation, programs):
        self.count += 1
        folder = self.root / "decisions" / f"{self.count:05d}"
        folder.mkdir(parents=True)
        state = folder / "state.pt"
        self.save_context(ctx, state)
        request = folder / "request.json"
        write_json(request, {
            "session_id": self.root.name, "decision": self.count,
            "step": ctx.step, "nfe": ctx.nfe, "sigma": ctx.sigma,
            "state": str(state), "workspace": str(programs.workspace.resolve()),
            "policy_version": programs.version, "observation": observation,
            "instructions": str((Path(__file__).parent / "prompts/sampling_agent.md").resolve()),
            "inspect_command": ["reactive-samplers", "inspect", str(state), "--workspace", str(programs.workspace.resolve())],
            "feedback": str(self.root / "feedback.json"),
        })
        started = time.perf_counter()
        try:
            answer = self.backend(request, programs.workspace.resolve())
            (folder / "answer.txt").write_text(answer or "", encoding="utf-8")
        finally:
            elapsed = time.perf_counter() - started
            self.seconds += elapsed
            write_json(folder / "timing.json", {"seconds": elapsed})
