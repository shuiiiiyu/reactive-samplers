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
        "def analyze(ctx, memory): return {'candidate': ctx.candidate_index}\n")
    (source / "updater.py").write_text(
        "from reactive_samplers.context import SkipAction\n"
        f"def choose(obs, ctx, memory): return SkipAction({rule})\n")
    return Programs.create(root / "workspace", root / "programs", source, method="grid_skip")


def run(root, programs, adapter=None, agent=None, budget=4):
    return sample(adapter or ToyAdapter(), programs, "fixture", 7, 4, budget,
                  CheckSchedule("checkpoints", (0,)), Recorder(root / "image", run_root=root),
                  agent, method="grid_skip")


class GridSkipTests(unittest.TestCase):
    def test_fixed_grid_control_and_recovery(self):
        for rule, budget, kept, skipped in (
            ("False", 4, [0, 1, 2, 3], []),
            ("ctx.candidate_index in (1, 3)", 4, [0, 2], [1, 3]),
            ("True", 4, [0], [1, 2, 3]),
            ("False", 2, [0, 1], [2, 3]),
        ):
            with self.subTest(rule=rule, budget=budget), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                programs = policy(root, rule)
                image, stats, _ = run(root, programs, budget=budget)
                self.assertEqual(image, 1.0)
                self.assertEqual(stats["nfe"], len(kept))
                self.assertEqual(stats["kept_indices"], kept)
                self.assertEqual(stats["skipped_indices"], skipped)
                self.assertTrue(stats["valid_for_benchmark"])
                self.assertTrue(set(stats["sigma_schedule"]).issubset(stats["reference_sigmas"]))
        for body in ("return Action(0.6)", "return SkipAction(1)",
                     "if ctx.candidate_index == 2: raise RuntimeError('mid-scan failure')\n"
                     "    return SkipAction(True)"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                programs = policy(root)
                (programs.workspace / "updater.py").write_text(
                    "from reactive_samplers.context import Action, SkipAction\n"
                    "def choose(obs, ctx, memory):\n    " + body + "\n")
                programs.reload()
                image, stats, _ = run(root, programs)
                self.assertEqual(image, 1.0)
                self.assertEqual(stats["nfe"], 4)
                self.assertEqual(stats["skipped_indices"], [])
                self.assertEqual(stats["default_fallback_steps"], 4)
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
                    "    if ctx.step == 1: raise RuntimeError('later failure')\n"
                    "    return SkipAction(ctx.candidate_index == 1)\n")
            agent = AgentSession(edit, root, lambda ctx, path: path.write_text("fixture"), method="grid_skip")
            image, stats, _ = run(root, programs, agent=agent)
            self.assertEqual(image, 1.0)
            self.assertEqual(stats["kept_indices"], [0, 2, 3])
            self.assertEqual(stats["skipped_indices"], [1])
            self.assertEqual(stats["nfe"], 3)
            self.assertEqual(requests[0]["method"], "grid_skip")
            self.assertEqual(requests[0]["candidate_index"], 1)
            self.assertEqual(Path(requests[0]["instructions"]).parts[-2:], ("grid_skip", "sampling_agent.md"))

    @unittest.skipIf(torch is None, "torch is needed for the initial template")
    def test_templates_freeze_and_sessions_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method in ("adaptive_sigma", "grid_skip"):
                programs = Programs.create(root / method, root / (method + "-versions"), method=method)
                self.assertEqual({path.relative_to(programs.workspace).as_posix() for path in source_files(programs.workspace)},
                                 {"analysis.py", "updater.py"})
            programs = policy(root)
            frozen = freeze(programs.versions / programs.version, root / "frozen", method="grid_skip")
            with self.assertRaises(ValueError):
                Programs.create(root / "wrong", root / "wrong-versions", frozen)
            replay = Programs.create(root / "replay", root / "replay-versions", frozen, method="grid_skip")
            self.assertEqual(replay.method, "grid_skip")
            first = state_directory(root / "auth", "/project/adaptive/agent_explore")
            second = state_directory(root / "auth", "/project/grid/agent_explore")
            self.assertNotEqual(first, second)
            legacy = root / "auth/reactive-samplers/agent_explore"
            (legacy / "decisions/00001").mkdir(parents=True)
            (legacy / "codex-session.json").write_text('{"thread_id": "old-session"}')
            (legacy / "decisions/00001/request.json").write_text(json.dumps({
                "state": "/project/adaptive/agent_explore/decisions/00001/state.pt"}))
            self.assertEqual(state_directory(root / "auth", "/project/adaptive/agent_explore"), legacy)
            self.assertNotEqual(state_directory(root / "auth", "/project/grid/agent_explore"), legacy)

    @unittest.skipIf(torch is None, "torch and diffusers are needed for numerical parity")
    def test_keep_all_matches_official_shifted_euler(self):
        from types import SimpleNamespace
        from diffusers import FlowMatchEulerDiscreteScheduler
        from reactive_samplers.context import Prediction
        from reactive_samplers.models.sd3 import SD3Adapter
        class Flow(ToyAdapter):
            def start(self, prompt, seed, steps):
                self.denoiser_items = 0
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

    @unittest.skipIf(torch is None, "torch is needed for feature hooks")
    def test_existing_hooks_and_temporal_features_need_no_probe_forward(self):
        from types import SimpleNamespace
        from reactive_samplers.context import Context
        from reactive_samplers.models.sd3 import SD3Adapter, save_context, load_context
        from reactive_samplers.prompts.grid_skip.analysis import analyze, FEATURE_LAYERS
        class Block(torch.nn.Module):
            def forward(self, hidden):
                return None, hidden + .01
        class Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pos_embed = torch.nn.Identity()
                self.transformer_blocks = torch.nn.ModuleList([Block() for _ in range(24)])
            def forward(self, hidden_states, **kwargs):
                value = self.pos_embed(hidden_states.flatten(2).transpose(1, 2))
                for block in self.transformer_blocks:
                    _, value = block(value)
                return (value.transpose(1, 2).reshape_as(hidden_states),)
        adapter = object.__new__(SD3Adapter)
        adapter.torch, adapter.device, adapter.cfg = torch, torch.device("cpu"), False
        adapter.embeds, adapter.pooled = None, None
        adapter.denoiser_items = adapter.denoiser_evaluations = 0
        adapter.pipe = SimpleNamespace(transformer=Transformer(), scheduler=SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000)))
        latent = torch.ones(1, 4, 4, 4)
        plain = adapter.predict(latent, 1., [])
        captured = adapter.predict(latent, 1., FEATURE_LAYERS)
        torch.testing.assert_close(plain.guided, captured.guided, rtol=0, atol=0)
        ctx = Context("test", 1, 0, 1, 4, 1., 0., 1000., (1., .75, .5, .25, 0.),
                      latent, captured, [], method="grid_skip", model_index=0,
                      candidate_index=1, candidate_timestep=750.)
        memory = {}
        first = analyze(ctx, memory)
        self.assertIsNotNone(first["feature_transform"])
        self.assertIsNone(first["input_feature_change"])
        ctx.candidate_index = 2
        ctx.candidate_timestep = 500.
        second = analyze(ctx, memory)
        self.assertEqual(first["feature_transform"], second["feature_transform"])
        self.assertEqual(second["grid"]["source_nfe"], 1)
        self.assertEqual(adapter.denoiser_evaluations, 2)
        ctx.nfe = 2
        self.assertIsNotNone(analyze(ctx, memory)["input_feature_change"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            save_context(ctx, path)
            self.assertEqual(load_context(path).candidate_index, 2)


if __name__ == "__main__":
    unittest.main()
