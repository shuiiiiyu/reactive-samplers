"""SD3-Medium only; official Diffusers forward, CFG, VAE and Euler step.

The adapter supplies the two physical sigma boundaries to the stock scheduler.
It never reimplements its numerical update or applies shifting a second time.
"""
import dataclasses
from pathlib import Path

from ..context import Prediction


class SD3Adapter:
    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        if type(pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise TypeError("Only the official FlowMatchEulerDiscreteScheduler is supported")
        if pipe.scheduler.config.get("stochastic_sampling", False) or pipe.scheduler.config.invert_sigmas:
            raise ValueError("The main experiment requires deterministic, descending FlowMatch Euler")
        self.torch, self.pipe = torch, pipe
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        self.max_sequence_length = max_sequence_length
        self.device = pipe._execution_device
        for model in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
            if model is not None:
                model.requires_grad_(False)
                model.eval()

    def start(self, prompt, seed, reference_steps):
        from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift
        torch, pipe = self.torch, self.pipe
        self.cfg = self.guidance_scale > 1
        with torch.inference_mode():
            pos, neg, pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, prompt_3=prompt,
                negative_prompt="", device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=self.cfg,
                max_sequence_length=self.max_sequence_length,
            )
            self.embeds = torch.cat([neg, pos], dim=0) if self.cfg else pos
            self.pooled = torch.cat([neg_pooled, pooled], dim=0) if self.cfg else pooled
            latent = pipe.prepare_latents(
                1, pipe.transformer.config.in_channels, self.height, self.width,
                pos.dtype, self.device, torch.Generator(device=self.device).manual_seed(seed), None,
            )
        kwargs = {}
        if pipe.scheduler.config.use_dynamic_shifting:
            h, w = latent.shape[-2:]
            config = pipe.scheduler.config
            kwargs["mu"] = calculate_shift(
                (h // pipe.transformer.config.patch_size) * (w // pipe.transformer.config.patch_size),
                config.base_image_seq_len, config.max_image_seq_len, config.base_shift, config.max_shift,
            )
        pipe.scheduler.set_timesteps(reference_steps, device=self.device, **kwargs)
        self.reference_sigmas = pipe.scheduler.sigmas.clone()
        sigmas = tuple(float(s) for s in self.reference_sigmas)
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError("Scheduler did not produce a strictly descending schedule")
        self.denoiser_items = 0
        self.denoiser_evaluations = 0
        self.synchronize()
        return latent, sigmas

    def timestep(self, sigma):
        return sigma * self.pipe.scheduler.config.num_train_timesteps

    def validate_feature_layers(self, feature_layers):
        modules = dict(self.pipe.transformer.named_modules())
        if not isinstance(feature_layers, (list, tuple)) or not all(isinstance(name, str) for name in feature_layers):
            raise TypeError("FEATURE_LAYERS must be a list or tuple of module names")
        if any(name not in modules for name in feature_layers):
            raise ValueError("FEATURE_LAYERS contains an unknown transformer module name")
        return modules

    def predict(self, latent, sigma, feature_layers):
        torch, pipe = self.torch, self.pipe
        features, hooks = {}, []
        modules = self.validate_feature_layers(feature_layers)
        for name in feature_layers:
            def capture(module, inputs, output, key=name):
                features[key] = output  # read-only contract; preserves arbitrary layer output structure
            hooks.append(modules[name].register_forward_hook(capture))
        try:
            with torch.inference_mode():
                model_input = torch.cat([latent] * 2) if self.cfg else latent
                t = torch.tensor(sigma, device=self.device, dtype=torch.float32)
                t = (t * pipe.scheduler.config.num_train_timesteps).expand(model_input.shape[0])
                self.denoiser_evaluations += 1
                self.denoiser_items += model_input.shape[0]
                output = pipe.transformer(
                    hidden_states=model_input, timestep=t, encoder_hidden_states=self.embeds,
                    pooled_projections=self.pooled, joint_attention_kwargs=None, return_dict=False,
                )[0]
                if self.cfg:
                    uncond, cond = output.chunk(2)
                    return Prediction(uncond + self.guidance_scale * (cond - uncond), cond, uncond, features)
                return Prediction(output, features=features)
        finally:
            for hook in hooks:
                hook.remove()

    def advance(self, latent, prediction, sigma, next_sigma):
        torch, scheduler = self.torch, self.pipe.scheduler
        # A one-interval view into the unchanged official Euler implementation.
        # These are already-shifted sigma values, not set_timesteps() inputs.
        scheduler.sigmas = torch.tensor([sigma, next_sigma], dtype=torch.float32, device="cpu")
        t = torch.tensor(sigma, device=self.device, dtype=torch.float32) * scheduler.config.num_train_timesteps
        scheduler.timesteps = t.reshape(1)
        scheduler._step_index = 0
        scheduler._begin_index = None
        with torch.inference_mode():
            return scheduler.step(prediction.guided, t, latent, return_dict=False)[0]

    def snapshot_state(self):
        from ..recovery import clone_state
        scheduler = self.pipe.scheduler
        return {key: clone_state(getattr(scheduler, key)) for key in
                ("sigmas", "timesteps", "_step_index", "_begin_index")}

    def restore_state(self, state):
        from ..recovery import clone_state
        for key, value in state.items():
            setattr(self.pipe.scheduler, key, clone_state(value))

    def save_failure_state(self, ctx, path):
        save_context(ctx, path)

    def finish(self, latent):
        torch, pipe = self.torch, self.pipe
        with torch.inference_mode():
            z = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            image = pipe.vae.decode(z, return_dict=False)[0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)


def save_context(ctx, path):
    """Persist a trusted local snapshot for repeated diagnostics while sampling pauses."""
    import torch
    def plain(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if dataclasses.is_dataclass(value):
            return {"__type__": type(value).__name__, **{f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}}
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    torch.save(plain(ctx), path)


def load_context(path):
    import torch
    from ..context import Context, Prediction
    def restore(value):
        if isinstance(value, dict):
            tag = value.get("__type__")
            fields = {k: restore(v) for k, v in value.items() if k != "__type__"}
            return {"Context": Context, "Prediction": Prediction}[tag](**fields) if tag else fields
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value
    return restore(torch.load(Path(path), map_location="cpu", weights_only=True))
