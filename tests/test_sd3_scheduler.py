"""Real Diffusers scheduler parity, without pretrained model weights."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"),
                     "install the sd3 extra to run official scheduler parity")
class SchedulerParity(unittest.TestCase):
    def test_stock_euler_and_dynamic_adapter_match(self):
        import torch
        from types import SimpleNamespace
        from diffusers import FlowMatchEulerDiscreteScheduler
        from reactive_samplers.context import Prediction
        from reactive_samplers.models.sd3 import SD3Adapter
        for shift in (1.0, 3.0):
            official = FlowMatchEulerDiscreteScheduler(shift=shift)
            official.set_timesteps(8)
            sigmas, times = official.sigmas.clone(), official.timesteps.clone()
            adapter = object.__new__(SD3Adapter)
            adapter.torch, adapter.device = torch, torch.device("cpu")
            adapter.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(shift=shift))
            expected = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(7))
            actual = expected.clone()
            for i, t in enumerate(times):
                velocity = expected.sin() + t / 1000
                expected = official.step(velocity, t, expected, return_dict=False)[0]
                actual = adapter.advance(actual, Prediction(velocity), float(sigmas[i]), float(sigmas[i + 1]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
