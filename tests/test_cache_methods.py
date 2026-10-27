"""CPU checks with real tiny Diffusers networks; no weights, GPU or paid Agent."""
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import torch
from diffusers import (EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler,
                       SD3Transformer2DModel, UNet2DConditionModel)

from reactive_samplers.agent import AgentSession
from reactive_samplers.context import Context, Prediction
from reactive_samplers.models.cache import prepare_feature, refresh_prediction, reuse_residual
from reactive_samplers.models.sd3 import SD3Adapter, save_context, load_context
from reactive_samplers.models.sdxl import SDXLAdapter
from reactive_samplers.programs import Programs, freeze
from reactive_samplers.report import Recorder
from reactive_samplers.schedule import CheckSchedule
from reactive_samplers.wrapper import sample


def adapter_fixture(model, cfg=True):
    torch.manual_seed(13)
    if model == "sd3":
        network = SD3Transformer2DModel(sample_size=8, patch_size=2, in_channels=4,
            out_channels=4, num_layers=2, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=16, caption_projection_dim=16, pooled_projection_dim=8,
            pos_embed_max_size=8).eval()
        adapter = object.__new__(SD3Adapter)
        adapter.pipe = SimpleNamespace(transformer=network, scheduler=FlowMatchEulerDiscreteScheduler(shift=3))
        adapter.pooled = torch.randn(2 if cfg else 1, 8)
    else:
        network = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            block_out_channels=(32, 64), layers_per_block=1, cross_attention_dim=16,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), attention_head_dim=4,
            norm_num_groups=8).eval()
        adapter = object.__new__(SDXLAdapter)
        adapter.pipe = SimpleNamespace(unet=network, scheduler=EulerDiscreteScheduler(beta_schedule="scaled_linear"))
        adapter.pooled, adapter.time_ids = torch.zeros(2 if cfg else 1, 4), torch.zeros(2 if cfg else 1, 6)
    network.requires_grad_(False)
    adapter.torch, adapter.device = torch, torch.device("cpu")
    adapter.cfg, adapter.guidance_scale = cfg, 5.
    adapter.embeds = torch.randn(2 if cfg else 1, 4, 16)
    initial = torch.randn(1, 4, 8, 8)
    def start(self, prompt, seed, steps):
        self.pipe.scheduler.set_timesteps(steps)
        sigmas = tuple(map(float, self.pipe.scheduler.sigmas))
        self.reference_times = dict(zip(sigmas[:-1], map(float, self.pipe.scheduler.timesteps)))
        self.reference_times[sigmas[-1]] = 0.
        self.generator = torch.Generator().manual_seed(seed)
        self.denoiser_items = self.denoiser_evaluations = 0
        return initial.clone(), sigmas
    adapter.start = MethodType(start, adapter)
    adapter.finish = lambda latent: latent
    return adapter, network


def programs_for(root, method, rule="False"):
    programs = Programs.create(root / "workspace", root / "programs", method=method)
    if rule != "False":
        (programs.workspace / "updater.py").write_text(
            "from reactive_samplers.context import SkipAction\n"
            f"def choose(obs, ctx, memory): return SkipAction({rule})\n")
        programs.reload()
    return programs


def run(root, method, adapter, programs, *, steps=6, budget=6, cap=2, agent=None):
    return sample(adapter, programs, "fixture", 7, steps, budget, CheckSchedule(),
                  Recorder(root / "trajectory", run_root=root), agent, method=method,
                  max_consecutive_skips=cap)


class CacheMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_keep_all_parity_and_no_extra_backbone_calls_both_models_cfg_modes(self):
        for model in ("sd3", "sdxl"):
            for cfg in (False, True):
                adapter, network = adapter_fixture(model, cfg)
                expected, sigmas = adapter.start("fixture", 7, 4)
                for sigma, target in zip(sigmas, sigmas[1:]):
                    expected = adapter.advance(expected, adapter.predict(expected, sigma, []), sigma, target)
                blocks = network.transformer_blocks if model == "sd3" else network.down_blocks
                for method in ("prediction_cache", "residual_cache"):
                    with self.subTest(model=model, cfg=cfg, method=method), tempfile.TemporaryDirectory() as tmp:
                        root, calls = Path(tmp), []
                        hooks = [block.register_forward_hook(lambda *args: calls.append(1)) for block in blocks]
                        try:
                            actual, stats, _ = run(root, method, adapter, programs_for(root, method), steps=4)
                        finally:
                            for hook in hooks:
                                hook.remove()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(stats["nfe"], 4)
                        self.assertEqual(stats["denoiser_items"], 4 * (2 if cfg else 1))
                        self.assertEqual(len(calls), 4 * len(blocks))
                        self.assertEqual(stats["scheduler_updates"], 4)
                        self.assertTrue(stats["valid_for_benchmark"])

    def test_reuse_skips_real_backbone_but_updates_every_grid_node_and_resets(self):
        for model in ("sd3", "sdxl"):
            for method in ("prediction_cache", "residual_cache"):
                with self.subTest(model=model, method=method), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    adapter, network = adapter_fixture(model)
                    programs = programs_for(root, method, "True")
                    blocks = network.transformer_blocks if model == "sd3" else network.down_blocks
                    calls = []
                    hooks = [block.register_forward_hook(lambda *args: calls.append(1)) for block in blocks]
                    try:
                        first, stats, _ = run(root, method, adapter, programs)
                        second, again, _ = run(root, method, adapter, programs)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                    self.assertEqual(stats["kept_indices"], [0, 3])
                    self.assertEqual(stats["reused_indices"], [1, 2, 4, 5])
                    self.assertEqual(stats["nfe"], again["nfe"])
                    self.assertEqual(stats["nfe"], 2)
                    self.assertEqual(len(calls), 2 * 2 * len(blocks))
                    self.assertEqual(stats["scheduler_updates"], 6)
                    self.assertEqual(stats["feature_extractions"], 6)
                    self.assertEqual(stats["output_head_evaluations"], 6 if method == "residual_cache" else 2)
                    self.assertEqual(stats["sigma_schedule"], stats["reference_sigmas"])

    def test_budget_reserves_mandatory_refresh_and_rejects_impossible_limit(self):
        for method in ("prediction_cache", "residual_cache"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                adapter, _ = adapter_fixture("sd3")
                programs = programs_for(root, method)
                _, stats, _ = run(root, method, adapter, programs, budget=2)
                self.assertEqual(stats["kept_indices"], [0, 3])
                with self.assertRaisesRegex(ValueError, "budget"):
                    run(root, method, adapter, programs, budget=1)

    def test_old_prediction_and_current_input_plus_old_residual_are_distinct(self):
        for model in ("sd3", "sdxl"):
            adapter, _ = adapter_fixture(model)
            latent, sigmas = adapter.start("fixture", 7, 4)
            feature = prepare_feature(adapter, latent, sigmas[0])
            real, residual = refresh_prediction(adapter, latent, sigmas[0], feature, residual=True)
            same = reuse_residual(adapter, latent, sigmas[0], feature, residual)
            torch.testing.assert_close(same.guided, real.guided, rtol=1e-5, atol=1e-6)
            later_feature = prepare_feature(adapter, latent + .1, sigmas[1])
            later = reuse_residual(adapter, latent + .1, sigmas[1], later_feature, residual)
            self.assertFalse(torch.equal(later.guided, real.guided))
            self.assertEqual(adapter.denoiser_evaluations, 1)

    def test_final_prediction_cache_matches_manual_scheduler_reuse(self):
        for model in ("sd3", "sdxl"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                adapter, _ = adapter_fixture(model)
                expected, sigmas = adapter.start("fixture", 7, 6)
                cached = None
                for step, (sigma, target) in enumerate(zip(sigmas, sigmas[1:])):
                    if step in (0, 3):
                        cached = adapter.predict(expected, sigma, [])
                    elif model == "sdxl":
                        # Official Euler expects input scaling at each node; its result
                        # is not used for a reused model output, but its state is valid.
                        prepare_feature(adapter, expected, sigma)
                    expected = adapter.advance(expected, cached, sigma, target)
                actual, _, _ = run(root, "prediction_cache", adapter,
                                    programs_for(root, "prediction_cache", "True"))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_entry_override_and_capture_hook_are_restored_after_failed_forward(self):
        for model in ("sd3", "sdxl"):
            adapter, network = adapter_fixture(model)
            latent, sigmas = adapter.start("fixture", 7, 4)
            feature = prepare_feature(adapter, latent, sigmas[0])
            entry = network.pos_embed if model == "sd3" else network.conv_in
            boundary = network.norm_out if model == "sd3" else network.conv_norm_out
            original_predict = adapter.predict
            hooks_before = len(boundary._forward_pre_hooks)
            def fail(*args):
                raise RuntimeError("forward failed")
            adapter.predict = fail
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                refresh_prediction(adapter, latent, sigmas[0], feature, residual=True)
            self.assertNotIn("forward", entry.__dict__)
            self.assertEqual(len(boundary._forward_pre_hooks), hooks_before)
            adapter.predict = original_predict
            self.assertIsNotNone(adapter.predict(latent, sigmas[0], []).guided)

    def test_pre_prediction_inspect_agent_reload_and_invalid_policy_recovery(self):
        for method in ("prediction_cache", "residual_cache"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                adapter, _ = adapter_fixture("sd3")
                programs = programs_for(root, method)
                snapshots = []
                def save(ctx, path):
                    self.assertIsNone(ctx.prediction.guided)
                    self.assertEqual(ctx.nfe, adapter.denoiser_evaluations)
                    save_context(ctx, path)
                    saved = load_context(path)
                    self.assertIsNone(saved.prediction.guided)
                    self.assertIsNotNone(saved.input_feature)
                    snapshots.append(ctx.step)
                def edit(request, workspace):
                    data = json.loads(request.read_text())
                    self.assertIn(method, data["instructions"])
                    self.assertEqual(set(data["observation"]), {"sigma", "step_delta_sigma",
                        "since_real_delta_sigma", "consecutive_skips", "has_reference", "feature_rel_change", "latent_rel_change"})
                    (workspace / "updater.py").write_text(
                        "from reactive_samplers.context import SkipAction\n"
                        "def choose(obs, ctx, memory): return SkipAction(True)\n")
                agent = AgentSession(edit, root, save, method=method)
                _, stats, _ = run(root, method, adapter, programs, agent=agent)
                self.assertEqual(snapshots, list(range(6)))
                self.assertEqual(stats["nfe"], 2)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                adapter, _ = adapter_fixture("sd3")
                programs = programs_for(root, method, "1")
                _, stats, _ = run(root, method, adapter, programs)
                self.assertEqual(stats["nfe"], 6)
                self.assertFalse(stats["valid_for_benchmark"])

    def test_frozen_methods_and_feature_reference_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            programs = programs_for(root, "prediction_cache")
            frozen = freeze(programs.versions / programs.version, root / "frozen", method="prediction_cache")
            with self.assertRaisesRegex(ValueError, "different sampling method"):
                Programs.create(root / "wrong", root / "wrong_versions", frozen, method="residual_cache")
        from reactive_samplers.prompts.prediction_cache.analysis import analyze
        ctx = Context("fixture", 7, 2, 1, 4, .5, 0., 500., (1., .5, 0.), None,
                      Prediction(None), [], input_feature=torch.tensor([[[1., -1.]]]),
                      reference_feature=torch.tensor([[[-1., 1.]]]),
                      cache_observation={"has_reference": True})
        # Mean pooling before distance would erase this spatial change entirely.
        self.assertEqual(analyze(ctx, {})["feature_rel_change"], 2.)


if __name__ == "__main__":
    unittest.main()
