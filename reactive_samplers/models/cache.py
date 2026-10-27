"""Cache boundaries for the pinned Diffusers 0.35.1 SD3/SDXL Base adapters.

Prediction cache stores the CFG-combined scheduler input. Residual cache stores
Z - H in hidden space, NOT a velocity/noise residual: SD3 pos_embed -> norm_out
input; SDXL conv_in -> conv_norm_out input (the whole down/mid/up U-Net).
Only refresh runs execute the denoising backbone. Reuse runs compute current H
and the model's original output head. No scheduler formulas are changed.
"""
from contextlib import contextmanager

from ..context import Prediction


def _sd3(adapter):
    return hasattr(adapter.pipe, "transformer")


def prepare_feature(adapter, latent, sigma):
    """Cheap current input feature before deciding whether to run the backbone.

    These starter features do not include timestep modulation. Sigma is supplied
    separately to the policy; this is not TeaCache's fitted decision estimator.
    SDXL must scale the CURRENT input, even when its prediction will be reused.
    """
    torch, pipe = adapter.torch, adapter.pipe
    with torch.inference_mode():
        value = torch.cat([latent, latent]) if adapter.cfg else latent
        if _sd3(adapter):
            return pipe.transformer.pos_embed(value).detach()
        adapter._interval(sigma, 0.0)
        value = pipe.scheduler.scale_model_input(value, pipe.scheduler.timesteps[0])
        if pipe.unet.config.center_input_sample:
            value = 2 * value - 1.0
        return pipe.unet.conv_in(value).detach()


@contextmanager
def _prepared_entry(module, feature):
    """Use the already computed entry feature on refresh; always restore forward.

    The adapter owns one paused inference stream, as in the existing sampler.
    Keep the stock forward for everything after this cheap entry module.
    """
    had_override = "forward" in module.__dict__
    original = module.forward
    module.forward = lambda *args, **kwargs: feature
    try:
        yield
    finally:
        if had_override:
            module.forward = original
        else:
            del module.forward


def refresh_prediction(adapter, latent, sigma, feature, *, residual):
    """Run the unchanged official forward; optionally cache its hidden-space delta."""
    pipe = adapter.pipe
    model = pipe.transformer if _sd3(adapter) else pipe.unet
    entry = model.pos_embed if _sd3(adapter) else model.conv_in
    boundary = model.norm_out if _sd3(adapter) else model.conv_norm_out
    if residual and boundary is None:
        raise ValueError("Residual caching requires the supported model's output normalization boundary")
    captured = []
    hook = boundary.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()) or None
    ) if residual else None
    try:
        with _prepared_entry(entry, feature):
            prediction = adapter.predict(latent, sigma, [])
    finally:
        if hook is not None:
            hook.remove()
    if not residual:
        return prediction, None
    if len(captured) != 1 or captured[0].shape != feature.shape:
        raise ValueError("Backbone input/output hidden shapes do not match the residual cache boundary")
    return prediction, (captured[0] - feature).detach()


def reuse_prediction(cached_prediction):
    """3. Zero-order hold: reuse the last real CFG-combined model prediction.

    In SD3 it is flow velocity; in the supported SDXL adapter it is epsilon.
    The scheduler converts/interprets it using the CURRENT latent and sigma.
    """
    return Prediction(cached_prediction.guided, cached_prediction.conditional,
                      cached_prediction.unconditional)


def reuse_residual(adapter, latent, sigma, feature, residual):
    """4. Current hidden input + old backbone delta -> CURRENT output head.

    SD3's output norm receives current timestep/text conditioning. SDXL's final
    norm/activation/convolution are unconditioned; its cached U-Net delta contains
    old time/text effects. This is an approximation, not full-model equivalence.
    """
    torch, pipe = adapter.torch, adapter.pipe
    with torch.inference_mode():
        hidden = feature + residual
        if _sd3(adapter):
            model = pipe.transformer
            timestep = torch.tensor(sigma, device=adapter.device, dtype=torch.float32)
            timestep = (timestep * pipe.scheduler.config.num_train_timesteps).expand(hidden.shape[0])
            temb = model.time_text_embed(timestep, adapter.pooled)
            output = model.proj_out(model.norm_out(hidden, temb))
            # Same unpatchify as the pinned official SD3 forward.
            patch = model.config.patch_size
            height, width = latent.shape[-2] // patch, latent.shape[-1] // patch
            output = output.reshape(output.shape[0], height, width, patch, patch, model.out_channels)
            output = torch.einsum("nhwpqc->nchpwq", output)
            output = output.reshape(output.shape[0], model.out_channels, height * patch, width * patch)
        else:
            model = pipe.unet
            if model.conv_norm_out is not None:
                hidden = model.conv_act(model.conv_norm_out(hidden))
            output = model.conv_out(hidden)
        if adapter.cfg:
            uncond, cond = output.chunk(2)
            return Prediction(uncond + adapter.guidance_scale * (cond - uncond), cond, uncond)
        return Prediction(output)
