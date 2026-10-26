"""Agent backends for local commands and the remote mailbox bridge."""
import json
import subprocess
import time
import os
import signal
from pathlib import Path

from ..programs import digest
from ..report import write_json


def terminate_process(process):
    """Terminate the CLI and its tool children, including on timeout or interruption."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=5)


class KeepAgent:
    def __call__(self, request_path, workspace):
        return "keep"


class CommandAgent:
    def __init__(self, argv, timeout_seconds=900):
        if not argv or not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("agent.command must be a nonempty argument list")
        self.argv = argv
        self.timeout_seconds = timeout_seconds

    def __call__(self, request_path, workspace):
        # The bridge owns CLI/provider session resumption. No shell interpolation.
        process = subprocess.Popen(
            [s.replace("{request}", str(request_path)).replace("{workspace}", str(workspace)) for s in self.argv],
            cwd=workspace, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        finally:
            terminate_process(process)
            process.stdout.close()
            process.stderr.close()
        if process.returncode:
            raise RuntimeError(f"Agent exited with status {process.returncode}: {stderr[-2000:]}")
        return stdout


class MailboxAgent:
    """Wait for a remote coding agent and serve same-step analysis requests."""
    def __init__(self, timeout_seconds=900, poll_seconds=1):
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("Mailbox timeout and poll interval must be positive")
        self.timeout_seconds, self.poll_seconds = timeout_seconds, poll_seconds

    def check(self, request_path, workspace, inspect):
        folder = Path(request_path).parent
        if (folder / "bridge_cancelled.json").exists():
            raise TimeoutError("This Agent request has already been cancelled")
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        response_path = folder / "bridge_response.json"
        deadline = time.monotonic() + self.timeout_seconds
        handled = set()
        while time.monotonic() < deadline:
            for path in sorted(folder.glob("inspect-*.request.json")):
                if path.name in handled:
                    continue
                handled.add(path.name)
                result_path = path.with_name(path.name.replace(".request.json", ".response.json"))
                try:
                    diagnostic = json.loads(path.read_text(encoding="utf-8"))
                    observation = inspect(diagnostic["workspace"])
                    result = {"status": "ok", "observation": observation}
                except (Exception, SystemExit) as exc:
                    result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                write_json(result_path, result)
            if response_path.is_file():
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if request.get("protocol_version") == 2:
                    if response.get("request_id") != request["request_id"] or response.get("base_version") != request["policy_version"]:
                        raise ValueError("Agent response belongs to a different request or policy version")
                    if digest(workspace) != request["policy_version"]:
                        raise ValueError("The running workspace changed while this request was pending")
                if response.get("status") != "ok":
                    raise RuntimeError(response.get("error", "Remote Agent bridge failed"))
                if request.get("protocol_version") == 2:
                    candidate = (folder / "candidate").resolve()
                    if Path(response.get("candidate_workspace", "")).resolve() != candidate:
                        raise ValueError("Agent candidate is outside this decision's staging directory")
                    if digest(candidate) != response.get("candidate_version"):
                        raise ValueError("Candidate hash does not match the published response")
                    return {"message": response.get("message", ""), "candidate_workspace": str(candidate),
                            "candidate_version": response["candidate_version"]}
                return response.get("message", "")
            time.sleep(self.poll_seconds)
        write_json(folder / "bridge_cancelled.json", {"reason": "sampling-side timeout",
                                                      "request_id": request.get("request_id")})
        raise TimeoutError(f"Agent bridge did not respond within {self.timeout_seconds} seconds")
