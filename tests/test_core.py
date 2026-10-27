"""CPU/std-library tests: no LLM calls, model downloads or generation experiments."""
import json
import tempfile
import unittest
from pathlib import Path

from reactive_samplers.agent import AgentSession
from reactive_samplers.cli import read_prompts
from reactive_samplers.context import Context, Prediction
from reactive_samplers.programs import Programs, copy_sources, digest, freeze
from reactive_samplers.report import Recorder
from reactive_samplers.schedule import CheckSchedule
from reactive_samplers.wrapper import sample


class ToyAdapter:
    """Scalar transport fixture, not an image model or a quality benchmark."""
    def start(self, prompt, seed, steps):
        import torch
        from types import SimpleNamespace
        class Entry(torch.nn.Module):
            def forward(self, value):
                return torch.as_tensor(value)
        self.torch, self.cfg = torch, False
        self.pipe = SimpleNamespace(transformer=SimpleNamespace(pos_embed=Entry(), norm_out=torch.nn.Identity()))
        self.denoiser_items = 0
        return 2.0, tuple(1 - i / steps for i in range(steps + 1))

    def predict(self, latent, sigma, feature_layers):
        self.denoiser_items += 1
        return Prediction(1.0)

    def timestep(self, sigma):
        return sigma * 1000

    def advance(self, latent, prediction, sigma, next_sigma):
        return latent + (next_sigma - sigma) * prediction.guided

    def finish(self, latent):
        return latent

    def synchronize(self):
        pass


def scalar_programs(workspace, versions):
    copy_sources(Path(__file__).resolve().parents[1] / "reactive_samplers/prompts/grid_skip", workspace)
    (workspace / "analysis.py").write_text("def analyze(ctx, memory):\n    return {'sigma': ctx.sigma}\n")
    return Programs(workspace, versions)


class CoreTests(unittest.TestCase):
    def test_default_rule_and_per_image_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            for _ in range(2):
                image, result, history = sample(ToyAdapter(), programs, "fixture", 0, 4, 4,
                                                CheckSchedule(), Recorder(root / "run"))
                self.assertEqual(image, 1.0)
                self.assertEqual(result["nfe"], 4)
                self.assertEqual(result["steps"], 4)
                self.assertEqual(result["timesteps"], [1000.0, 750.0, 500.0, 250.0])
                self.assertEqual(result["sigma_schedule"], [1.0, 0.75, 0.5, 0.25, 0.0])
                self.assertEqual(result["terminal_sigma"], 0)
                self.assertLessEqual(len(history), 2)

    def test_both_agent_schedules(self):
        self.assertTrue(all(CheckSchedule().due(i) for i in range(8)))
        schedule = CheckSchedule("checkpoints", (0, 3))
        self.assertEqual([i for i in range(8) if schedule.due(i)], [0, 3])
        with self.assertRaises(ValueError):
            CheckSchedule("checkpoints", (-1,))

    def test_agent_modifies_current_image_and_repeated_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            initial = programs.version
            calls = []
            def bridge(request_path, workspace):
                request = json.loads(request_path.read_text())
                calls.append((request["step"], request["nfe"]))
                # Both diagnostics happen inside one paused image, before its first update.
                (workspace / "analysis.py").write_text("def analyze(ctx, memory):\n    return {'target': 0.25}\n")
                diagnostic = Programs(workspace, root / "scratch_versions")
                self.assertEqual(diagnostic.analysis.analyze(None, {}), {"target": .25})
                (workspace / "analysis.py").write_text("def analyze(ctx, memory):\n    return {'target': 0.0}\n")
                (workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory):\n    return SkipAction(obs['target'] == 0.0)\n")
                return "Reached terminal in this same image"
            session = AgentSession(bridge, root, lambda ctx, path: path.write_text("fixture"))
            image, result, history = sample(ToyAdapter(), programs, "fixture", 0, 4, 4,
                                            CheckSchedule(), Recorder(root / "run"), session)
            self.assertEqual(calls, [(0, 0), (1, 1), (2, 1), (3, 1)])
            self.assertEqual(result["nfe"], 2)
            self.assertEqual(image, 1.5)
            self.assertNotEqual(initial, programs.version)
            self.assertEqual(history[0]["observation"]["target"], 0.)

    def test_invalid_edit_restores_old_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            initial = programs.version
            def bad(request, workspace):
                (workspace / "updater.py").write_text("def choose(:\n")
            session = AgentSession(bad, root, lambda ctx, path: path.write_text("fixture"))
            _, result, _ = sample(ToyAdapter(), programs, "fixture", 0, 2, 2,
                                   CheckSchedule("checkpoints", (0,)), Recorder(root / "run"), session)
            self.assertEqual(result["nfe"], 2)
            self.assertEqual(programs.version, initial)
            self.assertEqual(digest(programs.workspace), initial)
            self.assertIn("decision_rejected", (root / "run/events.jsonl").read_text())

    def test_illegal_candidate_action_rejected_before_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def bad(request, workspace):
                (workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory):\n    return SkipAction(1.2)\n")
            session = AgentSession(bad, root, lambda ctx, path: path.write_text("fixture"))
            image, _, _ = sample(ToyAdapter(), programs, "fixture", 0, 2, 2,
                                  CheckSchedule("checkpoints", (0,)), Recorder(root / "run"), session)
            self.assertEqual(image, 1.)

    def test_changed_updater_is_called_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            def edit(request, workspace):
                (workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory):\n"
                    "    memory['calls'] = memory.get('calls', 0) + 1\n"
                    "    assert memory['calls'] == ctx.step + 1\n"
                    "    return SkipAction(True)\n")
            session = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"))
            _, result, _ = sample(ToyAdapter(), programs, "fixture", 0, 2, 2,
                                   CheckSchedule(), Recorder(root / "run"), session)
            self.assertEqual(result["nfe"], 2)

    def test_freeze_verification_and_no_agent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            frozen = freeze(programs.versions / programs.version, root / "frozen")
            replay = Programs.create(root / "evaluation", root / "eval_versions", frozen)
            _, result, _ = sample(ToyAdapter(), replay, "fixture", 0, 3, 3,
                                   CheckSchedule(), Recorder(root / "run"))
            self.assertEqual(result["nfe"], 3)
            self.assertEqual(result["agent_seconds"], 0)
            (frozen / "analysis.py").write_text("def analyze(ctx, memory): return {}\n")
            with self.assertRaises(ValueError):
                Programs.create(root / "tampered", root / "bad_versions", frozen)

    def test_relative_helpers_reloaded_and_scratch_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = scalar_programs(root / "workspace", root / "programs")
            (programs.workspace / "signal.py").write_text("VALUE = 1\n")
            (programs.workspace / "analysis.py").write_text("from .signal import VALUE\ndef analyze(ctx, memory): return {'v': VALUE}\n")
            programs.reload()
            self.assertEqual(programs.analysis.analyze(None, {}), {"v": 1})
            (programs.workspace / "signal.py").write_text("VALUE = 2\n")
            programs.reload()
            self.assertEqual(programs.analysis.analyze(None, {}), {"v": 2})
            version = programs.version
            (programs.workspace / "scratch").mkdir()
            (programs.workspace / "scratch/x.py").write_text("invalid syntax !")
            self.assertFalse(programs.reload())
            self.assertEqual(programs.version, version)

    def test_prompt_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text('{"id":"a","prompt":"fixture","seed":0}\n')
            self.assertEqual(len(read_prompts(path, 1)), 1)
            with self.assertRaises(ValueError):
                read_prompts(path, 10)


if __name__ == "__main__":
    unittest.main()
