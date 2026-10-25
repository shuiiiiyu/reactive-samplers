"""Full official-pipeline parity gate; requires local SD3 weights and explicit config."""
import tempfile
from pathlib import Path

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="sd3_main")
def main(cfg):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    from reactive_samplers.models.sd3 import SD3Adapter
    from reactive_samplers.programs import Programs
    from reactive_samplers.report import Recorder, write_json
    from reactive_samplers.schedule import CheckSchedule
    from reactive_samplers.wrapper import sample

    # The parity prompt is a technical fixture, never an exploration/benchmark record.
    prompt, seed = "A red cube on a wooden table", 0
    pipe = StableDiffusion3Pipeline.from_pretrained(
        cfg.model_path, torch_dtype=getattr(torch, cfg.dtype), local_files_only=True,
    ).to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    scheduler_config = dict(pipe.scheduler.config)
    expected_steps, actual_steps = [], []
    def capture(pipe, step, timestep, kwargs):
        expected_steps.append(kwargs["latents"].detach().cpu().clone())
        return kwargs

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    expected = pipe(
        prompt=prompt, negative_prompt="", height=cfg.height, width=cfg.width,
        guidance_scale=cfg.guidance_scale, num_inference_steps=cfg.reference_steps,
        max_sequence_length=cfg.max_sequence_length,
        generator=torch.Generator(device=cfg.device).manual_seed(seed),
        output_type="latent", callback_on_step_end=capture,
    ).images.detach().cpu()

    class TracedAdapter(SD3Adapter):
        def advance(self, *args):
            latent = super().advance(*args)
            actual_steps.append(latent.detach().cpu().clone())
            return latent

        def finish(self, latent):
            return latent.detach().cpu()

    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    adapter = TracedAdapter(pipe, guidance_scale=cfg.guidance_scale, height=cfg.height,
                            width=cfg.width, max_sequence_length=cfg.max_sequence_length)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        programs = Programs.create(root / "workspace", root / "programs")
        actual, stats, _ = sample(adapter, programs, prompt, seed, cfg.reference_steps,
                                   cfg.reference_steps, CheckSchedule(), Recorder(root / "trace"))
    if len(expected_steps) != len(actual_steps):
        raise AssertionError("The adapter executed a different number of denoising steps")
    errors = []
    for i, (a, b) in enumerate(zip(expected_steps, actual_steps)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4, msg=f"latent parity failed at step {i}")
        errors.append(float((a - b).abs().max()))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    report = {"passed": True, "nfe": stats["nfe"], "max_abs_error_per_step": errors,
              "rtol": 1e-4, "atol": 1e-4, "model_path": cfg.model_path}
    destination = Path("sampler_runs/parity/sd3_pipeline.json")
    write_json(destination, report)
    print(destination.resolve())


if __name__ == "__main__":
    main()
