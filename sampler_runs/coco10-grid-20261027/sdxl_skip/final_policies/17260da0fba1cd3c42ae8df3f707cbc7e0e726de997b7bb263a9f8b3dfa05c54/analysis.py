"""SDXL epsilon observations plus input/final UNet hidden feature changes."""
from reactive_samplers.prompts.grid_skip.analysis import analyze as grid_analysis

FEATURE_LAYERS = ['conv_in', 'conv_norm_out']
HISTORY_LIMIT = 2


def analyze(ctx, memory):
    observation = grid_analysis(ctx, memory, 'conv_in', 'conv_norm_out')
    observation['model'] = 'sdxl'
    observation['prediction_type'] = 'epsilon'
    observation['noise_prediction'] = observation.pop('velocity')
    observation['noise_prediction_change'] = observation.pop('velocity_change')
    return observation
