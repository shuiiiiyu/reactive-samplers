"""Opt-in GPU smoke: one image per official pipeline, never run by unittest discover.

Use tests/run_inference_smoke.sh inside tmux on the configured RunPod image.
This checks checkpoint/inference health, not the adaptive method's quality or speed.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback


def main():
    import numpy as np
    import torch
    import diffusers
    import transformers
    from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", "sd3", "sdxl"], default="both")
    parser.add_argument("--prompt", default="A red panda wearing a blue scarf, sitting on a wooden bench in a sunlit park, detailed photography.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("sampler_runs/smoke_generation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires the RunPod CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    specs = {
        "sd3": (StableDiffusion3Pipeline, "sd3-medium", torch.bfloat16, 28, 7.0),
        "sdxl": (StableDiffusionXLPipeline, "sdxl-base-1.0", torch.float16, 30, 5.0),
    }
    results = {"prompt": args.prompt, "seed": args.seed,
               "environment": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                               "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
               "images": {}}
    def record():
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    record()
    failed = False
    for name in specs if args.model == "both" else [args.model]:
        cls, folder, dtype, steps, cfg = specs[name]
        path = Path("models") / folder
        pipe, hook, denoiser = None, None, None
        entry = {"status": "loading", "checkpoint": str(path.resolve()), "dtype": str(dtype),
                 "steps": steps, "guidance_scale": cfg, "height": 1024, "width": 1024,
                 "cpu_offload": True, "local_files_only": True}
        results["images"][name] = entry
        record()
        print(json.dumps({"model": name, "status": "loading"}), flush=True)
        try:
            load_start = time.perf_counter()
            pipe = cls.from_pretrained(path, torch_dtype=dtype, local_files_only=True, use_safetensors=True)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            entry["load_seconds"] = time.perf_counter() - load_start
            counts = {"nfe": 0, "executed_steps": 0}
            denoiser = pipe.transformer if name == "sd3" else pipe.unet
            def forward_count(module, inputs, output):
                counts["nfe"] += 1
            hook = denoiser.register_forward_hook(forward_count)
            def check_latents(pipeline, step, timestep, kwargs):
                if not torch.isfinite(kwargs["latents"]).all().item():
                    raise FloatingPointError(f"Nonfinite latents at step {step}")
                counts["executed_steps"] += 1
                print(json.dumps({"model": name, "step": step + 1, "total": steps}), flush=True)
                return kwargs
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = pipe(
                prompt=args.prompt, negative_prompt="", num_inference_steps=steps,
                guidance_scale=cfg, height=1024, width=1024,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                callback_on_step_end=check_latents,
            )
            torch.cuda.synchronize()
            entry["inference_seconds"] = time.perf_counter() - start
            image = output.images[0]
            pixels = np.asarray(image)
            if image.size != (1024, 1024) or not np.isfinite(pixels).all() or float(pixels.std()) < 1.0:
                raise ValueError("Output image is invalid or effectively blank")
            if counts["nfe"] != steps or counts["executed_steps"] != steps:
                raise ValueError("Unexpected denoiser/step count")
            destination = args.output_dir / f"{name}.png"
            image.save(destination)
            from PIL import Image
            with Image.open(destination) as saved:
                saved.verify()
            entry.update(status="complete", image=str(destination.resolve()), **counts,
                         pixel_std=float(pixels.std()), pixel_min=int(pixels.min()), pixel_max=int(pixels.max()),
                         peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            print(json.dumps({"model": name, **entry}), flush=True)
            del output, image, pixels
        except Exception as exc:
            failed = True
            entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
        finally:
            if hook is not None:
                hook.remove()
            if pipe is not None:
                pipe.maybe_free_model_hooks()
            del pipe
            del denoiser
            gc.collect()
            torch.cuda.empty_cache()
            record()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
