"""Real UNet/Euler CPU checks; full pretrained pipeline parity is a GPU run gate."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel
from reactive_samplers.models.sdxl import SDXLAdapter
from reactive_samplers.agent import AgentSession
from reactive_samplers.context import Context, Prediction
from reactive_samplers.prompts.sdxl_grid_skip.analysis import analyze, FEATURE_LAYERS


class SDXLGridTests(unittest.TestCase):
    def test_official_unet_euler_scaling_skips_features_and_restoration(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)
        unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            block_out_channels=(32, 64), layers_per_block=1, cross_attention_dim=16,
            down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D', 'UpBlock2D'), attention_head_dim=4,
            norm_num_groups=8).eval()
        for indices in ([0, 1, 2, 3], [0, 2, 3]):
            official = EulerDiscreteScheduler(beta_schedule='scaled_linear')
            official.set_timesteps(4)
            sigmas, times = official.sigmas.clone(), official.timesteps.clone()
            official.sigmas = torch.cat([sigmas[indices], sigmas[-1:]])
            official.timesteps = times[indices]
            adapter = object.__new__(SDXLAdapter)
            adapter.torch, adapter.device = torch, torch.device('cpu')
            adapter.pipe = SimpleNamespace(unet=unet, scheduler=EulerDiscreteScheduler(beta_schedule='scaled_linear'))
            adapter.cfg, adapter.guidance_scale = True, 5.
            adapter.embeds = torch.randn(2, 4, 16)
            adapter.pooled, adapter.time_ids = torch.zeros(2, 4), torch.zeros(2, 6)
            adapter.reference_times = dict(zip(map(float, sigmas[:-1]), map(float, times)))
            adapter.reference_times[0.] = 0.
            adapter.generator = torch.Generator().manual_seed(7)
            generator = torch.Generator().manual_seed(7)
            adapter.denoiser_items = adapter.denoiser_evaluations = 0
            actual = torch.randn(1, 4, 8, 8)
            expected = actual.clone()
            with torch.inference_mode():
                for position, index in enumerate(indices):
                    sigma = float(sigmas[index])
                    target = float(sigmas[indices[position + 1]]) if position + 1 < len(indices) else 0.
                    model_input = official.scale_model_input(torch.cat([expected, expected]), times[index])
                    raw = unet(model_input, times[index], encoder_hidden_states=adapter.embeds, return_dict=False)[0]
                    uncond, cond = raw.chunk(2)
                    expected = official.step(uncond + 5 * (cond - uncond), times[index], expected,
                                             generator=generator, return_dict=False)[0]
                    prediction = adapter.predict(actual, sigma, FEATURE_LAYERS)
                    snapshot = adapter.snapshot_state()
                    candidate = adapter.advance(actual, prediction, sigma, target)
                    adapter.restore_state(snapshot)
                    actual = adapter.advance(actual, prediction, sigma, target)
                    torch.testing.assert_close(actual, candidate, rtol=0, atol=0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(adapter.denoiser_evaluations, len(indices))
            ctx = Context('fixture', 7, 0, 1, 4, sigma, 0., float(times[index]), tuple(map(float, sigmas)),
                          actual, prediction, [], method='grid_skip', model_index=0,
                          candidate_index=1, candidate_timestep=float(times[1]))
            memory = {}
            first = analyze(ctx, memory)
            self.assertEqual(first['prediction_type'], 'epsilon')
            self.assertIsNotNone(first['feature_transform'])
            self.assertNotIn('velocity', first)
            ctx.nfe = 2
            self.assertIsNotNone(analyze(ctx, memory)['output_feature_change'])

    def test_sdxl_gets_its_own_agent_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = AgentSession(None, Path(folder), None, method='grid_skip', model='sdxl')
            self.assertEqual(agent.instructions.parent.name, 'sdxl_grid_skip')


if __name__ == '__main__':
    unittest.main()
