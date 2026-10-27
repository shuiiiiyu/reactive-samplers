"""One CPU check for fixed-grid control, recovery, features and method isolation."""
import json
from pathlib import Path
import tempfile
import unittest

from test_core import ToyAdapter
from reactive_samplers.agent import AgentSession
from reactive_samplers.programs import Programs, freeze, source_files
from reactive_samplers.report import Recorder
from reactive_samplers.schedule import CheckSchedule
from reactive_samplers.wrapper import sample
from scripts.runpod_codex_bridge import state_directory

try:
    import torch
except ImportError:
    torch = None


def policy(root, rule="False"):
    source = root / "starter"
    source.mkdir()
    (source / "analysis.py").write_text(
        "def analyze(ctx, memory): return {'node': ctx.step}\n")
    (source / "updater.py").write_text(
        "from reactive_samplers.context import SkipAction\n"
        f"def choose(obs, ctx, memory): return SkipAction({rule})\n")
    return Programs.create(root / "workspace", root / "programs", source, method="grid_skip")


def run(root, programs, adapter=None, agent=None, budget=4):
    return sample(adapter or ToyAdapter(), programs, "fixture", 7, 4, budget,
                  CheckSchedule("checkpoints", (0,)), Recorder(root / "image", run_root=root),
                  agent, method="grid_skip")


class GridSkipTests(unittest.TestCase):
    def test_custom_sigma_method_is_unavailable(self):
        from reactive_samplers.context import METHODS
        from reactive_samplers import wrapper
        self.assertEqual(METHODS, ("grid_skip", "prediction_cache", "residual_cache"))
        self.assertFalse(hasattr(wrapper, "sample_adaptive_sigma"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unknown sampling method"):
                Programs.create(root / "workspace", root / "versions", method="adaptive_sigma")

    def test_fixed_grid_control_and_recovery(self):
        for rule, budget, kept, skipped in (
            ("False", 4, [0, 1, 2, 3], []),
            ("ctx.step == 1", 4, [0, 2, 3], [1]),
            ("True", 4, [0, 3], [1, 2]),
            ("False", 2, [0, 3], [1, 2]),
        ):
            with self.subTest(rule=rule, budget=budget), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                programs = policy(root, rule)
                image, stats, _ = run(root, programs, budget=budget)
                self.assertEqual(image, 2.0 - len(kept)/4)
                self.assertEqual(stats["nfe"], len(kept))
                self.assertEqual(stats["kept_indices"], kept)
                self.assertEqual(stats["skipped_indices"], skipped)
                self.assertTrue(stats["valid_for_benchmark"])
                self.assertTrue(set(stats["sigma_schedule"]).issubset(stats["reference_sigmas"]))
        for body in ("return None", "return SkipAction(1)",
                     "if ctx.step == 2: raise RuntimeError('current-node failure')\n"
                     "    return SkipAction(True)"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                programs = policy(root)
                (programs.workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory):\n    " + body + "\n")
                programs.reload()
                image, stats, _ = run(root, programs)
                later_failure = "current-node failure" in body
                self.assertEqual(image, 1.25 if later_failure else 1.0)
                self.assertEqual(stats["nfe"], 3 if later_failure else 4)
                self.assertEqual(stats["skipped_indices"], [1] if later_failure else [])
                self.assertEqual(stats["default_fallback_steps"], 2 if later_failure else 4)
                self.assertFalse(stats["valid_for_benchmark"])

    def test_agent_switch_and_later_failure_keep_committed_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = policy(root)
            requests = []
            def edit(path, workspace):
                requests.append(json.loads(path.read_text()))
                (workspace / "updater.py").write_text(
                    "from reactive_samplers.context import SkipAction\n"
                    "def choose(obs, ctx, memory):\n"
                    "    if ctx.step == 2: raise RuntimeError('later failure')\n"
                    "    return SkipAction(ctx.step == 1)\n")
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"), method="grid_skip")
            image, stats, _ = run(root, programs, agent=agent)
            self.assertEqual(image, 1.25)
            self.assertEqual(stats["kept_indices"], [0, 2, 3])
            self.assertEqual(stats["skipped_indices"], [1])
            self.assertEqual(stats["nfe"], 3)
            self.assertEqual(requests[0]["method"], "grid_skip")
            self.assertEqual(requests[0]["decision_context"]["node_index"], 0)
            self.assertEqual(Path(requests[0]["instructions"]).parts[-2:], ("grid_skip", "sampling_agent.md"))

    @unittest.skipIf(torch is None, "torch is needed for the initial template")
    def test_templates_freeze_and_sessions_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in ("grid_skip", "prediction_cache", "residual_cache"):
                programs = Programs.create(root / method, root / (method + "-versions"), method=method)
                self.assertEqual({path.relative_to(programs.workspace).as_posix() for path in source_files(programs.workspace)},
                                 {"analysis.py", "updater.py"})
            programs = policy(root)
            frozen = freeze(programs.versions / programs.version, root / "frozen", method="grid_skip")
            with self.assertRaises(ValueError):
                Programs.create(root / "wrong", root / "wrong-versions", frozen, method="prediction_cache")
            replay = Programs.create(root / "replay", root / "replay-versions", frozen, method="grid_skip")
            self.assertEqual(replay.method, "grid_skip")
            first = state_directory(root / "auth", "/project/adaptive/agent_explore")
            second = state_directory(root / "auth", "/project/grid/agent_explore")
            self.assertNotEqual(first, second)

    @unittest.skipIf(torch is None, "torch and diffusers are needed for numerical parity")
    def test_keep_all_matches_official_shifted_euler(self):
        from types import SimpleNamespace
        from diffusers import FlowMatchEulerDiscreteScheduler
        from reactive_samplers.context import Prediction
        from reactive_samplers.models.sd3 import SD3Adapter
        class Flow(ToyAdapter):
            def start(self, prompt, seed, steps):
                super().start(prompt, seed, steps)
                self.solver = object.__new__(SD3Adapter)
                self.solver.torch, self.solver.device = torch, torch.device("cpu")
                scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
                scheduler.set_timesteps(steps)
                self.solver.pipe = SimpleNamespace(scheduler=scheduler)
                return torch.ones(1, 16, 4, 4), tuple(map(float, scheduler.sigmas))
            def predict(self, latent, sigma, feature_layers):
                self.denoiser_items += 1
                return Prediction(latent.sin() + torch.tensor(sigma) * 1000 / 1000)
            def advance(self, latent, prediction, sigma, next_sigma):
                return self.solver.advance(latent, prediction, sigma, next_sigma)
            def snapshot_state(self):
                return self.solver.snapshot_state()
            def restore_state(self, state):
                self.solver.restore_state(state)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual, stats, _ = run(root, policy(root), Flow())
            official = FlowMatchEulerDiscreteScheduler(shift=3.0)
            official.set_timesteps(4)
            expected = torch.ones_like(actual)
            for timestep in official.timesteps:
                expected = official.step(expected.sin() + timestep / 1000, timestep, expected, return_dict=False)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(stats["nfe"], 4)



if __name__ == "__main__":
    unittest.main()
