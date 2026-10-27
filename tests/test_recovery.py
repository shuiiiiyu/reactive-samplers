"""Fault injection: no model weights, external services, or Agent charges."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import unittest
try:
    import torch
except ImportError:
    torch = None

from test_core import ToyAdapter, scalar_programs
from reactive_samplers.agent import AgentSession
from reactive_samplers.backends import CommandAgent, MailboxAgent
from reactive_samplers.context import Context, Prediction
from reactive_samplers.programs import copy_sources, digest
from reactive_samplers.report import Recorder, summarize, write_json
from reactive_samplers.schedule import CheckSchedule
from reactive_samplers.wrapper import sample
from scripts.runpod_codex_bridge import parse_turn, policy_digest


FOLLOW = ("from reactive_samplers.context import SkipAction\n"
          "def choose(obs, ctx, memory):\n"
          "    return SkipAction(False)\n")


class RecoveryTests(unittest.TestCase):
    def run_image(self, root, programs, agent=None, adapter=None, checkpoints=(0,), steps=4, **kwargs):
        return sample(adapter or ToyAdapter(), programs, "fixture", 7, steps, steps,
                      CheckSchedule("checkpoints", checkpoints),
                      Recorder(root / "trajectories/image", run_root=root), agent, **kwargs)

    @unittest.skipUnless(os.name == "posix", "Linux policy watchdog")
    def test_python_policy_loop_times_out_and_reuses_the_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def edit(path, workspace):
                (workspace / "updater.py").write_text("def choose(obs, ctx, memory):\n    while True: pass\n")
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = self.run_image(root, programs, agent, policy_timeout_seconds=.03)
            self.assertEqual(image, 1.0)
            self.assertEqual(stats["nfe"], 4)
            self.assertIn("Policy execution exceeded", (root / "failure_report.md").read_text())

    @unittest.skipUnless(os.name == "posix", "Linux policy import watchdog")
    def test_python_import_loop_does_not_replace_the_active_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original = programs.version
            (programs.workspace / "updater.py").write_text("while True: pass\n")
            with self.assertRaises(TimeoutError):
                programs.reload(timeout_seconds=.03)
            self.assertEqual(programs.version, original)
            programs.restore(original)

    def test_late_runtime_failure_is_sent_to_next_agent_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original = programs.version
            requests = []
            def edit(path, workspace):
                request = json.loads(path.read_text())
                requests.append(request)
                if len(requests) == 1:
                    (workspace / "updater.py").write_text(FOLLOW.replace(
                        "    return SkipAction", "    if ctx.step == 1:\n        memory['poison'] = True\n"
                        "        raise RuntimeError('late failure')\n    return SkipAction"))
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = self.run_image(root, programs, agent, checkpoints=(0, 2))
            self.assertEqual(image, 1.0)
            self.assertEqual(stats["nfe"], 4)
            self.assertEqual(stats["recovery_steps"], 1)
            self.assertEqual(programs.version, original)
            self.assertEqual(requests[1]["previous_answer"]["outcome"], "reverted")
            self.assertTrue(any(row["error"] == "late failure" for row in requests[1]["failures"]))
            rows = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()]
            self.assertTrue(rows[0]["continued"])
            self.assertEqual(rows[0]["policy_memory"], "reset")
            self.assertFalse(stats["valid_for_benchmark"])
            self.assertIn("late failure", (root / "failure_report.md").read_text())

    def test_diagnostic_failure_keeps_the_active_rule_and_is_reported_next_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original, requests = programs.version, []
            test = self
            class Diagnose:
                def check(self, path, workspace, inspect):
                    request = json.loads(path.read_text())
                    requests.append(request)
                    if len(requests) == 1:
                        candidate = Path(request["diagnostic_workspace"])
                        copy_sources(workspace, candidate)
                        (candidate / "analysis.py").write_text(
                            "def analyze(ctx, memory):\n    memory['bad'] = True\n"
                            "    raise ValueError('diagnostic only')\n")
                        with test.assertRaisesRegex(ValueError, "diagnostic only"):
                            inspect(candidate)
                    return "keep"
            agent = AgentSession(Diagnose(), root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = self.run_image(root, programs, agent, checkpoints=(0, 2))
            self.assertEqual(image, 1.0)
            self.assertEqual(programs.version, original)
            self.assertEqual(stats["recovery_steps"], 0)
            self.assertEqual(requests[1]["failures"][0]["outcome"], "diagnostic_error")
            self.assertTrue(requests[1]["failures"][0]["continued"])

    def test_pending_trial_is_not_promoted_to_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original, candidates = programs.version, []
            def edit(path, workspace):
                step = json.loads(path.read_text())["step"]
                code = FOLLOW + f"\n# candidate {step}\n"
                if step == 1:
                    code = code.replace("    return SkipAction", "    if ctx.step == 2:\n"
                                        "        raise RuntimeError('trial B failed')\n    return SkipAction")
                (workspace / "updater.py").write_text(code)
                candidates.append(digest(workspace))
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            _, stats, _ = self.run_image(root, programs, agent, checkpoints=(0, 1))
            self.assertEqual(programs.version, original)
            self.assertTrue(all(version not in programs.recovery_versions for version in candidates))
            self.assertEqual(stats["nfe"], 4)

    def test_fallback_that_also_raises_reaches_earlier_stable_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original = programs.version
            def edit(path, workspace):
                step = json.loads(path.read_text())["step"]
                message = "A" if step == 0 else "B"
                (workspace / "updater.py").write_text(FOLLOW.replace(
                    "    return SkipAction", f"    if ctx.step == 3:\n        raise RuntimeError('{message}')\n    return SkipAction"))
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = self.run_image(root, programs, agent, checkpoints=(0, 2), steps=8)
            self.assertEqual(image, 1.0)
            self.assertEqual(programs.version, original)
            self.assertEqual(stats["failure_count"], 2)

    def test_corrupt_stable_version_is_skipped_with_an_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            original = programs.version
            def edit(path, workspace):
                step = json.loads(path.read_text())["step"]
                if step == 0:
                    (workspace / "updater.py").write_text(FOLLOW + "\n# A\n")
                else:
                    stable = programs.versions / programs.version / "analysis.py"
                    stable.write_text(stable.read_text() + "\n# tampered immutable version\n")
                    (workspace / "updater.py").write_text(FOLLOW.replace(
                        "    return SkipAction", "    if ctx.step == 3:\n        raise RuntimeError('B failed')\n    return SkipAction"))
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = self.run_image(root, programs, agent, checkpoints=(0, 2), steps=8)
            self.assertEqual(image, 1.0)
            self.assertEqual(programs.version, original)
            failures = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["stage"] == "restore_version" for row in failures))
            self.assertEqual(stats["nfe"], 8)

    def test_running_analysis_systemexit_and_rng_changes_are_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def edit(path, workspace):
                (workspace / "analysis.py").write_text(
                    "import random\ndef analyze(ctx, memory):\n"
                    "    if ctx.step == 1:\n        random.random()\n"
                    "        ctx.history[0]['latent'] = 999\n        memory['poison'] = True\n"
                    "        raise SystemExit('analysis failed')\n    return {}\n")
            state = random.getstate()
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, history = self.run_image(root, programs, agent)
            self.assertEqual(image, 1.0)
            self.assertEqual(history[0]["latent"], 2.0)
            self.assertEqual(random.getstate(), state)
            self.assertEqual(stats["failure_count"], 1)

    def test_invalid_candidate_import_observation_and_coordinates(self):
        candidates = [
            ("analysis.py", "raise SystemExit('import failed')\n"),
            ("analysis.py", "def analyze(ctx, memory): return {'bad': float('nan')}\n"),
            ("analysis.py", "FEATURE_LAYERS = 17\ndef analyze(ctx, memory): return {}\n"),
            ("updater.py", "from reactive_samplers.context import SkipAction\n"
             "def choose(obs, ctx, memory):\n    ctx.sigma = 2.0\n    return SkipAction(1.5)\n"),
        ]
        for name, code in candidates:
            with self.subTest(name=name, code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                programs = scalar_programs(root / "workspace", root / "programs")
                original = programs.version
                def edit(path, workspace):
                    (workspace / name).write_text(code)
                agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
                image, stats, _ = self.run_image(root, programs, agent)
                self.assertEqual(image, 1.0)
                self.assertEqual(programs.version, original)
                self.assertEqual(stats["nfe"], 4)
                self.assertEqual(json.loads((root / "decisions/00001/outcome.json").read_text())["outcome"], "rejected")

    def test_solver_partial_state_and_nan_do_not_commit_or_add_nfe(self):
        class Solver(ToyAdapter):
            state = 0
            failed = False
            def snapshot_state(self):
                return self.state
            def restore_state(self, state):
                self.state = state
            def advance(self, latent, prediction, sigma, next_sigma):
                self.state += 1
                if not self.failed and sigma == 1.0:
                    self.failed = True
                    return float("nan")
                return super().advance(latent, prediction, sigma, next_sigma)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def edit(path, workspace):
                (workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\ndef choose(obs, ctx, memory): return SkipAction(False) # revised rule\n")
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            adapter = Solver()
            image, stats, history = self.run_image(root, programs, agent, adapter)
            self.assertEqual(image, 1.0)
            self.assertEqual(adapter.state, 4)
            self.assertEqual(stats["nfe"], 4)
            self.assertEqual(stats["retry_nfe"], 0)
            self.assertEqual(len(history), 4)

    def test_frozen_failure_is_labeled_when_default_takes_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            (programs.workspace / "updater.py").write_text("def choose(obs, ctx, memory): raise RuntimeError('bad frozen rule')\n")
            programs.reload()
            image, stats, _ = self.run_image(root, programs)
            self.assertEqual(image, 1.0)
            self.assertEqual(stats["default_fallback_steps"], 4)
            self.assertFalse(stats["valid_for_benchmark"])
            self.assertEqual(summarize([stats])["valid_count"], 0)

    def test_cuda_failure_aborts_with_snapshot_without_policy_retry(self):
        class Broken(ToyAdapter):
            snapshots = 0
            def predict(self, *args):
                self.denoiser_items += 1
                raise RuntimeError("CUDA error: illegal memory access")
            def save_failure_state(self, ctx, path):
                self.snapshots += 1
                path.write_text("trusted fixture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            adapter = Broken()
            with self.assertRaisesRegex(RuntimeError, "CUDA error"):
                self.run_image(root, programs, adapter=adapter)
            self.assertEqual(adapter.denoiser_items, 1)
            self.assertEqual(adapter.snapshots, 1)
            report = json.loads((root / "trajectories/image/failure.json").read_text())
            self.assertEqual(report["status"], "aborted")
            self.assertTrue(report["state_saved"])

    @unittest.skipIf(torch is None, "torch is required for tensor isolation")
    def test_policy_cannot_mutate_authoritative_latent_prediction_or_history(self):
        class Tensors(ToyAdapter):
            def start(self, prompt, seed, steps):
                _, sigmas = super().start(prompt, seed, steps)
                with torch.inference_mode():
                    return torch.full((1, 1, 2, 2), 2.0), sigmas
            def predict(self, latent, sigma, feature_layers):
                self.denoiser_items += 1
                return Prediction(torch.ones_like(latent))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def edit(path, workspace):
                (workspace / "updater.py").write_text(FOLLOW.replace(
                    "    return SkipAction", "    ctx.latent.fill_(999)\n    ctx.input_feature.fill_(888)\n"
                    "    if ctx.history:\n        ctx.history[0]['latent'].zero_()\n"
                    "    if ctx.step == 1:\n        memory['poison'] = ctx.latent\n"
                    "        raise RuntimeError('tensor mutation before failure')\n    return SkipAction"))
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            image, stats, history = self.run_image(root, programs, agent, Tensors())
            torch.testing.assert_close(image, torch.ones_like(image), rtol=0, atol=0)
            torch.testing.assert_close(history[0]["latent"], torch.full_like(image, 2.0), rtol=0, atol=0)
            self.assertEqual(stats["nfe"], 4)


class MailboxRecoveryTests(unittest.TestCase):
    def prepare(self, root):
        programs = scalar_programs(root / "workspace", root / "programs")
        folder = root / "decisions/00001"
        folder.mkdir(parents=True)
        request = {"protocol_version": 2, "request_id": "unique-request", "policy_version": programs.version}
        write_json(folder / "request.json", request)
        copy_sources(programs.workspace, folder / "candidate")
        response = {"status": "ok", "request_id": request["request_id"], "base_version": programs.version,
                    "candidate_workspace": str(folder / "candidate"), "candidate_version": digest(folder / "candidate")}
        return programs, folder, response

    def test_staged_candidate_is_returned_without_touching_active_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            programs, folder, response = self.prepare(Path(tmp))
            (folder / "candidate/updater.py").write_text(FOLLOW + "\n# new version\n")
            response["candidate_version"] = digest(folder / "candidate")
            write_json(folder / "bridge_response.json", response)
            before = digest(programs.workspace)
            result = MailboxAgent(1, .001).check(folder / "request.json", programs.workspace, None)
            self.assertEqual(result["candidate_workspace"], str((folder / "candidate").resolve()))
            self.assertEqual(digest(programs.workspace), before)

    def test_protocol_two_mailbox_commits_a_valid_candidate_at_the_same_step(self):
        class Staged(MailboxAgent):
            def check(self, path, workspace, inspect):
                request = json.loads(Path(path).read_text())
                candidate = Path(path).parent / "candidate"
                copy_sources(workspace, candidate)
                (candidate / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory): return SkipAction(True)\n")
                write_json(Path(path).parent / "bridge_response.json", {
                    "status": "ok", "request_id": request["request_id"], "base_version": request["policy_version"],
                    "candidate_workspace": str(candidate), "candidate_version": digest(candidate),
                })
                return super().check(path, workspace, inspect)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            initial = programs.version
            agent = AgentSession(Staged(1, .001), root, lambda ctx, path: path.write_text("fixture"))
            image, stats, _ = sample(ToyAdapter(), programs, "fixture", 0, 4, 4, CheckSchedule(),
                                     Recorder(root / "image", run_root=root), agent)
            self.assertEqual(image, 1.5)
            self.assertEqual(stats["nfe"], 2)
            self.assertNotEqual(programs.version, initial)
            self.assertEqual(digest(programs.workspace), programs.version)

    def test_wrong_request_hash_and_late_response_are_rejected(self):
        for fault in ("request_id", "cancelled", "changed_base"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                programs, folder, response = self.prepare(Path(tmp))
                if fault in response:
                    response[fault] = "wrong"
                elif fault == "cancelled":
                    with self.assertRaises(TimeoutError):
                        MailboxAgent(.01, .001).check(folder / "request.json", programs.workspace, None)
                else:
                    (programs.workspace / "updater.py").write_text(FOLLOW + "# changed base\n")
                write_json(folder / "bridge_response.json", response)
                with self.assertRaises((ValueError, TimeoutError)):
                    MailboxAgent(1, .001).check(folder / "request.json", programs.workspace, None)

    def test_sampler_rejects_a_changed_staged_candidate(self):
        class Staged(MailboxAgent):
            def check(self, path, workspace, inspect):
                request = json.loads(Path(path).read_text())
                candidate = Path(path).parent / "candidate"
                copy_sources(workspace, candidate)
                (candidate / "updater.py").write_text(FOLLOW + "# valid proposed version\n")
                proposed = digest(candidate)
                (candidate / "updater.py").write_text(FOLLOW + "# modified after publication\n")
                write_json(Path(path).parent / "bridge_response.json", {
                    "status": "ok", "request_id": request["request_id"], "base_version": request["policy_version"],
                    "candidate_workspace": str(candidate), "candidate_version": proposed,
                })
                return super().check(path, workspace, inspect)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            initial = programs.version
            agent = AgentSession(Staged(1, .001), root, lambda ctx, path: path.write_text("fixture"))
            _, stats, _ = sample(ToyAdapter(), programs, "fixture", 0, 4, 4,
                CheckSchedule("checkpoints", (0,)), Recorder(root / "image", run_root=root), agent)
            self.assertEqual(programs.version, initial)
            self.assertEqual(stats["nfe"], 4)
            self.assertEqual(json.loads((root / "decisions/00001/outcome.json").read_text())["outcome"], "rejected")

    def test_incomplete_and_failed_cli_turns_are_not_successes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cli.jsonl"
            start = {"type": "thread.started", "thread_id": "session"}
            for events in ([start], [start, {"type": "turn.failed"}],
                           [start, {"type": "turn.completed"}, {"type": "error"}],
                           [start, {"type": "item.completed", "item": {"type": "error",
                            "message": "Code Mode unavailable: host executable was not found"}},
                            {"type": "turn.completed"}]):
                path.write_text("\n".join(json.dumps(event) for event in events))
                with self.assertRaises(RuntimeError):
                    parse_turn(path)
            path.write_text(json.dumps(start) + "\n" + json.dumps({"type": "turn.completed"}))
            self.assertEqual(parse_turn(path)[0], "session")
            with self.assertRaises(RuntimeError):
                parse_turn(path, "different-session")

    @unittest.skipUnless(os.name == "posix", "experiment CLI process groups run on Linux")
    def test_command_timeout_terminates_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = ("import subprocess,sys; from pathlib import Path; "
                      "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                      "Path('child.pid').write_text(str(child.pid))")
            with self.assertRaises(Exception) as raised:
                CommandAgent([sys.executable, "-c", script], .5)(root / "request.json", root)
            self.assertIn("Timeout", type(raised.exception).__name__)
            child = int((root / "child.pid").read_text())
            process = Path(f"/proc/{child}/stat")
            deadline = time.monotonic() + 1
            while process.exists() and process.read_text().split()[2] != "Z" and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(not process.exists() or process.read_text().split()[2] == "Z")

    def test_bridge_hash_matches_sampler_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            programs = scalar_programs(Path(tmp) / "workspace", Path(tmp) / "programs")
            self.assertEqual(policy_digest(programs.workspace), digest(programs.workspace))


if __name__ == "__main__":
    unittest.main()
