import json
import unittest

try:
    import torch
except ImportError:
    torch = None

from reactive_samplers.context import Context, Prediction


@unittest.skipIf(torch is None, "torch is required for online observation checks")
class AnalysisTests(unittest.TestCase):
    def test_flow_clean_spatial_cfg_and_layer_signals(self):
        from reactive_samplers.prompts.analysis import analyze
        latent = torch.ones(1, 4, 16, 16)
        velocity = torch.full_like(latent, 0.25)
        previous = {"latent": latent * 1.1, "sigma": 0.9,
                    "prediction": Prediction(velocity * 1.1)}
        prediction = Prediction(velocity, velocity, velocity * 0.5,
                                {"block": (latent, None)})
        ctx = Context("test", 1, 1, 2, 4, 0.8, 0.0, 800.0, (0.9, 0.8, 0.0),
                      latent, prediction, [previous])
        observation = analyze(ctx, {})
        self.assertAlmostEqual(observation["predicted_clean"]["mean"], 0.8, places=6)
        self.assertAlmostEqual(observation["predicted_clean_change"]["rms"], 0.0525, places=6)
        self.assertAlmostEqual(observation["cfg"]["difference"]["rms"], 0.125, places=6)
        self.assertEqual(len(observation["spatial_predicted_clean_change"]["tile_rms_4x4"]), 16)
        self.assertEqual(observation["layer_features"]["block"][0]["shape"], [1, 4, 16, 16])
        json.dumps(observation, allow_nan=False)
        ctx.history = []
        self.assertIsNone(analyze(ctx, {})["predicted_clean_change"])


if __name__ == "__main__":
    unittest.main()
