"""Fixed-grid SDXL adapter using the unchanged Diffusers Euler scheduler."""
from ..sd3 import SD3Adapter
from ...context import Prediction


class SDXLAdapter(SD3Adapter):
    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length=None):
        import torch
        from diffusers import EulerDiscreteScheduler
        if type(pipe.scheduler) is not EulerDiscreteScheduler or pipe.scheduler.config.prediction_type != 'epsilon':
            raise ValueError('SDXL grid skipping requires official Euler with epsilon prediction')
        if pipe.unet.config.time_cond_proj_dim is not None:
            raise ValueError('This adapter targets SDXL Base without guidance embeddings')
        self.torch, self.pipe, self.device = torch, pipe, pipe._execution_device
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        for model in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
            if model is not None:
                model.requires_grad_(False)
                model.eval()

    def start(self, prompt, seed, reference_steps):
        torch, pipe = self.torch, self.pipe
        self.cfg = self.guidance_scale > 1
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            pos, neg, pooled, neg_pooled = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, negative_prompt='', device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=self.cfg)
            pipe.scheduler.set_timesteps(reference_steps, device=self.device)
            latent = pipe.prepare_latents(1, pipe.unet.config.in_channels, self.height, self.width,
                                         pos.dtype, self.device, self.generator, None)
            projection = pipe.text_encoder_2.config.projection_dim if pipe.text_encoder_2 else pooled.shape[-1]
            times = pipe._get_add_time_ids((self.height, self.width), (0, 0), (self.height, self.width),
                                           dtype=pos.dtype, text_encoder_projection_dim=projection).to(self.device)
            self.embeds = torch.cat([neg, pos]) if self.cfg else pos
            self.pooled = torch.cat([neg_pooled, pooled]) if self.cfg else pooled
            self.time_ids = torch.cat([times, times]) if self.cfg else times
        self.reference_sigmas = pipe.scheduler.sigmas.clone()
        sigmas = tuple(map(float, self.reference_sigmas))
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError('SDXL reference sigmas must strictly descend')
        self.reference_times = dict(zip(sigmas[:-1], map(float, pipe.scheduler.timesteps)))
        self.reference_times[sigmas[-1]] = 0.0  # Terminal boundary has no model evaluation.
        self.denoiser_items = self.denoiser_evaluations = 0
        self.synchronize()
        return latent, sigmas

    def timestep(self, sigma):
        return self.reference_times[sigma]  # No interpolation: this adapter only removes grid nodes.

    def _interval(self, sigma, next_sigma):
        torch, scheduler = self.torch, self.pipe.scheduler
        scheduler.sigmas = torch.tensor([sigma, next_sigma], dtype=torch.float32)
        scheduler.timesteps = torch.tensor([self.timestep(sigma)], device=self.device, dtype=torch.float32)
        scheduler._step_index, scheduler._begin_index = 0, None

    def validate_feature_layers(self, layers):
        modules = dict(self.pipe.unet.named_modules())
        if any(name not in modules for name in layers):
            raise ValueError('FEATURE_LAYERS contains an unknown UNet module')
        return modules

    def predict(self, latent, sigma, feature_layers):
        torch, pipe = self.torch, self.pipe
        self._interval(sigma, 0.0)
        features, hooks = {}, []
        modules = self.validate_feature_layers(feature_layers)
        for name in feature_layers:
            def capture(module, inputs, output, key=name):
                features[key] = output
            hooks.append(modules[name].register_forward_hook(capture))
        try:
            with torch.inference_mode():
                model_input = torch.cat([latent, latent]) if self.cfg else latent
                timestep = pipe.scheduler.timesteps[0]
                model_input = pipe.scheduler.scale_model_input(model_input, timestep)
                self.denoiser_evaluations += 1
                self.denoiser_items += model_input.shape[0]
                output = pipe.unet(model_input, timestep, encoder_hidden_states=self.embeds,
                                   added_cond_kwargs={'text_embeds': self.pooled, 'time_ids': self.time_ids},
                                   return_dict=False)[0]
                if self.cfg:
                    uncond, cond = output.chunk(2)
                    return Prediction(uncond + self.guidance_scale * (cond - uncond), cond, uncond, features)
                return Prediction(output, features=features)
        finally:
            for hook in hooks:
                hook.remove()

    def advance(self, latent, prediction, sigma, next_sigma):
        self._interval(sigma, next_sigma)
        with self.torch.inference_mode():
            return self.pipe.scheduler.step(prediction.guided, self.pipe.scheduler.timesteps[0], latent,
                                             generator=self.generator, return_dict=False)[0]

    def snapshot_state(self):
        return {**super().snapshot_state(), 'is_scale_input_called': self.pipe.scheduler.is_scale_input_called,
                '_generator_state': self.generator.get_state()}

    def restore_state(self, state):
        self.generator.set_state(state['_generator_state'])
        super().restore_state({key: value for key, value in state.items() if key != '_generator_state'})

    def finish(self, latent):
        torch, pipe = self.torch, self.pipe
        with torch.inference_mode():
            upcast = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
            if upcast:
                pipe.upcast_vae()
                latent = latent.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
            mean, std = pipe.vae.config.get('latents_mean'), pipe.vae.config.get('latents_std')
            if mean is not None and std is not None:
                latent = latent * torch.tensor(std, device=latent.device, dtype=latent.dtype).view(1, 4, 1, 1) / pipe.vae.config.scaling_factor + torch.tensor(mean, device=latent.device, dtype=latent.dtype).view(1, 4, 1, 1)
            else:
                latent = latent / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(latent, return_dict=False)[0]
            if upcast:
                pipe.vae.to(dtype=torch.float16)
            if pipe.watermark is not None:
                image = pipe.watermark.apply_watermark(image)
            return pipe.image_processor.postprocess(image, output_type='pil')[0]


def verify_pipeline(pipe, adapter, prompt, seed, steps):
    """Full real-weight no-skip parity gate, excluded from measured sample timings."""
    latent, sigmas = adapter.start(prompt, seed, steps)
    for sigma, next_sigma in zip(sigmas, sigmas[1:]):
        latent = adapter.advance(latent, adapter.predict(latent, sigma, []), sigma, next_sigma)
    expected = pipe(prompt=prompt, negative_prompt='', num_inference_steps=steps,
                    guidance_scale=adapter.guidance_scale, height=adapter.height, width=adapter.width,
                    generator=adapter.torch.Generator(device=adapter.device).manual_seed(seed),
                    output_type='latent').images
    adapter.torch.testing.assert_close(latent, expected, rtol=0, atol=0)
    return {'status': 'passed', 'max_abs_latent_error': float((latent - expected).abs().max()),
            'steps': steps, 'seed': seed, 'rtol': 0, 'atol': 0}
