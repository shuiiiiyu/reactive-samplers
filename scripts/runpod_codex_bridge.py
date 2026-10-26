"""Relay a Trento sampling mailbox to one resumable, workspace-limited Codex session."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import signal
import time


def stop_process(process):
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


def policy_digest(workspace):
    hashed = hashlib.sha256()
    for path in sorted(Path(workspace).rglob("*.py")):
        if "scratch" in path.relative_to(workspace).parts or ".agent_tools" in path.relative_to(workspace).parts:
            continue
        hashed.update(path.relative_to(workspace).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return hashed.hexdigest()


def state_directory(codex_home, run_dir):
    root = Path(codex_home) / "reactive-samplers"
    legacy = root / Path(run_dir).name
    # Keep a legacy conversation only when all its saved requests belong to this run.
    if (legacy / "codex-session.json").is_file():
        try:
            owners = {json.loads(path.read_text(encoding="utf-8")).get("state", "").rsplit("/decisions/", 1)[0]
                      for path in (legacy / "decisions").glob("*/request.json")}
        except (OSError, ValueError):
            owners = set()
        if owners == {run_dir}:
            return legacy
    return root / (Path(run_dir).name + "-" + hashlib.sha256(run_dir.encode()).hexdigest()[:16])


def parse_turn(path, previous_thread=None):
    thread_id, message, completed = previous_thread, "", False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"turn.failed", "error"}:
            raise RuntimeError(f"Codex reported a failed turn; see {path}")
        incoming = event.get("thread_id")
        if incoming:
            if thread_id and incoming != thread_id:
                raise RuntimeError("Codex resumed a different conversation")
            thread_id = incoming
        completed |= event.get("type") == "turn.completed"
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            message = item.get("text", message)
    if not thread_id or not completed:
        raise RuntimeError(f"Codex returned an incomplete turn; see {path}")
    return thread_id, message


def run(argv, timeout=60, capture=True):
    return subprocess.run(argv, check=True, text=True, capture_output=capture, timeout=timeout)


def ssh(host, command, timeout=60):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], timeout)


def exists(host, path):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, "test", "-f", path]).returncode == 0


def rsync_policy(host, source, destination, *, pull=False):
    source, destination = str(source), str(destination)
    remote = f"{host}:{source.rstrip('/')}/"
    local = source.rstrip("/") + "/"
    remote_destination = f"{host}:{destination.rstrip('/')}/"
    local_destination = destination.rstrip("/") + "/"
    argv = ["rsync", "-a", "--delete", "--exclude=scratch/", "--exclude=.agent_tools/",
            "--include=*/", "--include=*.py", "--exclude=*", "-e", "ssh"]
    argv += [remote, local_destination] if pull else [local, remote_destination]
    run(argv, timeout=180)


def put_json(host, remote_path, value):
    local = Path("/tmp") / f"reactive-samplers-{os.getpid()}-{Path(remote_path).name}"
    temporary = local.with_suffix(local.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    run(["rsync", "-a", "-e", "ssh", str(temporary), f"{host}:{remote_path}.tmp"])
    ssh(host, f"mv {shlex.quote(remote_path + '.tmp')} {shlex.quote(remote_path)}")
    temporary.unlink(missing_ok=True)


def relay_inspections(args, workspace, decision_dir, remote_diagnostic, decision_deadline=None):
    helper_dir = workspace / ".agent_tools"
    for marker in sorted(helper_dir.glob("inspect-*.request.json")):
        local_response = marker.with_name(marker.name.replace(".request.json", ".response.json"))
        if local_response.exists():
            continue
        requested_workspace = json.loads(marker.read_text(encoding="utf-8")).get("workspace")
        if Path(requested_workspace or "").resolve() != workspace.resolve():
            raise ValueError("Inspect helper requested a workspace outside the policy copy")
        rsync_policy(args.trento_host, workspace, remote_diagnostic)
        remote_request = f"{decision_dir}/{marker.name}"
        remote_response = f"{decision_dir}/{local_response.name}"
        put_json(args.trento_host, remote_request, {"workspace": remote_diagnostic})
        deadline = time.monotonic() + args.agent_timeout
        if decision_deadline is not None:
            deadline = min(deadline, decision_deadline)
        while time.monotonic() < deadline and not exists(args.trento_host, remote_response):
            if exists(args.trento_host, decision_dir + "/bridge_cancelled.json"):
                raise TimeoutError("Trento cancelled this Agent decision")
            time.sleep(1)
        if not exists(args.trento_host, remote_response):
            raise TimeoutError("Timed out waiting for the Trento inspection")
        temporary = local_response.with_suffix(local_response.suffix + ".tmp")
        run(["rsync", "-a", "-e", "ssh", f"{args.trento_host}:{remote_response}", str(temporary)])
        temporary.replace(local_response)


def process_decision(args, request_path):
    host = args.trento_host
    decision_dir = str(Path(request_path).parent).replace("\\", "/")
    if not decision_dir.startswith(args.run_dir + "/decisions/"):
        raise ValueError("Decision path is outside the selected run")
    local_decision = args.state_dir / "decisions" / Path(decision_dir).name
    local_decision.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "-e", "ssh", f"{host}:{decision_dir}/request.json", str(local_decision / "request.json")])
    request_file = local_decision / "request.json"
    request = json.loads(request_file.read_text(encoding="utf-8"))
    remote_workspace = args.run_dir + "/workspace"
    remote_diagnostic = decision_dir + "/diagnostic_workspace"
    if request.get("workspace") != remote_workspace or request.get("diagnostic_workspace") != remote_diagnostic:
        raise ValueError("Request workspace does not match this run")
    method = request.get("method", "adaptive_sigma")
    model = request.get("model", "sd3")
    prompt_name = {("sd3", "adaptive_sigma"): "sampling_agent.md",
                   ("sd3", "grid_skip"): "grid_skip/sampling_agent.md",
                   ("sdxl", "grid_skip"): "sdxl_grid_skip/sampling_agent.md"}.get((model, method))
    if prompt_name is None or request.get("instructions") != args.project_root + "/reactive_samplers/prompts/" + prompt_name:
        raise ValueError("Unexpected Agent instruction path")
    if request.get("feedback") != args.run_dir + "/feedback.json":
        raise ValueError("Unexpected feedback path")

    workspace = args.state_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    rsync_policy(host, remote_workspace, workspace, pull=True)
    instructions = local_decision / "sampling_agent.md"
    run(["rsync", "-a", "-e", "ssh", f"{host}:{request['instructions']}", str(instructions)])
    helper = args.bridge_root / "scripts" / "agent_inspect_helper.py"
    if not helper.is_file():
        raise FileNotFoundError(helper)
    local_helper = workspace / ".agent_tools" / "inspect_current"
    local_helper.parent.mkdir(exist_ok=True)
    local_helper.write_bytes(helper.read_bytes())
    request["request"] = str(request_file)
    request["workspace"] = str(workspace)
    request["instructions"] = str(instructions)
    request["inspect_command"] = ["python3", ".agent_tools/inspect_current"]
    if exists(host, request["feedback"]):
        local_feedback = local_decision / "feedback.json"
        run(["rsync", "-a", "-e", "ssh", f"{host}:{request['feedback']}", str(local_feedback)])
        request["feedback"] = str(local_feedback)
    for key, name in (("failure_report", "failure_report.md"), ("failure_log", "failures.jsonl")):
        remote = request.get(key)
        if remote:
            if remote != args.run_dir + "/" + name:
                raise ValueError("Unexpected failure report path")
            if exists(host, remote):
                local = local_decision / name
                run(["rsync", "-a", "-e", "ssh", f"{host}:{remote}", str(local)])
                request[key] = str(local)
            else:
                request[key] = None
    request["state_access"] = "State stays on Trento; use inspect_command for same-step analysis."
    request_file.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    session_file = args.state_dir / "codex-session.json"
    session = json.loads(session_file.read_text(encoding="utf-8")) if session_file.exists() else {}
    if session and session.get("method", "adaptive_sigma") != method:
        raise ValueError("Different sampling methods require independent Agent runs and conversations")
    if session and session.get("model", "sd3") != model:
        raise ValueError("Different base models require independent Agent conversations")
    prompt = (f"Read {instructions} and {request_file}. Edit only analysis.py, updater.py, and relative Python "
              f"helpers in {workspace}. You may repeatedly edit analysis.py and run `python3 .agent_tools/inspect_current` "
              "to inspect the exact paused state. Then edit or keep updater.py. Do not alter the model, sampler, "
              "scheduler, evaluation, prompts, seeds, or other project files. End with a concise rationale and prediction.")
    command = [args.codex, "exec", "--json", "-m", "gpt-6-luna",
               "-c", "model_reasoning_effort=" + args.reasoning_effort,
               "-c", 'service_tier="default"', "-c", "features.fast_mode=false", "-C", str(workspace),
               "--sandbox", "danger-full-access", "--skip-git-repo-check"]
    if session.get("thread_id"):
        command += ["resume", session["thread_id"]]
    command.append(prompt)
    env = os.environ.copy()
    env["CODEX_HOME"] = str(args.codex_home)
    log_path = local_decision / "codex.jsonl"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > args.agent_timeout:
                    raise TimeoutError("Codex CLI decision timed out")
                if exists(host, decision_dir + "/bridge_cancelled.json"):
                    raise TimeoutError("Sampler cancelled this decision")
                relay_inspections(args, workspace, decision_dir, remote_diagnostic,
                                  started + args.agent_timeout)
                time.sleep(0.25)
        finally:
            stop_process(process)
    if process.returncode:
        raise RuntimeError(f"Codex CLI exited with status {process.returncode}; see {log_path}")
    thread_id, final_message = parse_turn(log_path, session.get("thread_id"))
    session_file.write_text(json.dumps({"thread_id": thread_id, "method": method, "model": model}) + "\n", encoding="utf-8")
    if exists(host, decision_dir + "/bridge_cancelled.json"):
        return
    # Protocol 2 never writes a running workspace. Only the sampler may accept a staged candidate.
    staged = decision_dir + "/candidate"
    target = staged if request.get("protocol_version") == 2 else remote_workspace
    rsync_policy(host, workspace, target)
    if exists(host, decision_dir + "/bridge_cancelled.json"):
        return
    put_json(host, decision_dir + "/bridge_response.json", {
        "status": "ok", "message": final_message or "Codex CLI turn completed.",
        "agent_model": "gpt-6-luna", "reasoning_effort": args.reasoning_effort,
        "service_tier": "default", "fast_mode": False, "sandbox_mode": "danger-full-access",
        "request_id": request.get("request_id"), "base_version": request["policy_version"],
        "candidate_workspace": staged, "candidate_version": policy_digest(workspace),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--trento-host", default="trento")
    parser.add_argument("--codex", default="/tmp/codex-v0.160.1")
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex-opal"))
    parser.add_argument("--bridge-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--poll-seconds", type=float, default=1)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    args = parser.parse_args()
    args.run_dir = args.run_dir.rstrip("/")
    args.project_root = args.project_root.rstrip("/")
    if not args.run_dir.startswith(args.project_root + "/sampler_runs/"):
        parser.error("--run-dir must be under project_root/sampler_runs/")
    args.codex_home = Path(args.codex_home).expanduser().resolve()
    args.bridge_root = Path(args.bridge_root).resolve()
    args.state_dir = state_directory(args.codex_home, args.run_dir)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.codex).is_file():
        parser.error("Codex CLI missing; run scripts/bootstrap_runpod_codex.sh first")
    print(f"Watching {args.run_dir}; one Codex session is reused for this run.", flush=True)
    while True:
        result = ssh(args.trento_host, f"find {shlex.quote(args.run_dir + '/decisions')} -mindepth 2 -maxdepth 2 -name request.json -print 2>/dev/null || true")
        for request_path in sorted(result.stdout.splitlines()):
            request_path = request_path.strip()
            if not request_path or exists(args.trento_host, str(Path(request_path).parent / "bridge_response.json")):
                continue
            try:
                process_decision(args, request_path)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                decision = str(Path(request_path).parent).replace("\\", "/")
                saved = args.state_dir / "decisions" / Path(decision).name / "request.json"
                request = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
                # Keep incomplete edits for diagnosis, but never publish them as an accepted candidate.
                workspace = args.state_dir / "workspace"
                if workspace.is_dir():
                    try:
                        rsync_policy(args.trento_host, workspace, decision + "/rejected")
                    except Exception:
                        pass
                put_json(args.trento_host, decision + "/bridge_response.json", {
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "request_id": request.get("request_id"), "base_version": request.get("policy_version"),
                })
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
