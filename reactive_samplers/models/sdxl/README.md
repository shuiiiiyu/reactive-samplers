# SDXL fixed-grid adapter

SDXL Base uses official EulerDiscreteScheduler with epsilon prediction. The adapter
only removes original grid nodes, preserving model input scaling, CFG, time IDs,
VAE upcasting and watermarking. Continuous sigma actions are not supported.
CPU parity tests cover scaling/steps; a configured real-weight pipeline parity gate
must pass before each small-experiment arm is measured.
