"""Editable starting probe. The observation dictionary is intentionally unrestricted."""

FEATURE_LAYERS = []  # names in SD3 transformer's named_modules(); captured on subsequent forwards
HISTORY_LIMIT = 2    # frozen inference: retain this many past states; None requests full history


def analyze(ctx, memory):
    # Add tensor/feature diagnostics here. No extra denoiser evaluation is necessary.
    return {
        "sigma": ctx.sigma,
        "timestep": ctx.timestep,
        "nfe": ctx.nfe,
        "remaining_nfe": ctx.remaining_nfe,
        "history_length": len(ctx.history),
    }
