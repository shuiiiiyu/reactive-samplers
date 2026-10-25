# Current status

- Framework: within-image Agent checks, repeated diagnostics, hot reload, versions and freeze.
- CPU checks: 11 passed, including exact tested official FlowMatch Euler interval parity.
- Base models: official SD3-Medium and SDXL Base 1.0 downloaded locally.
- Inference smoke: both official pipelines generated a valid 1024x1024 image.
- SD3: sampler_runs/smoke_generation_sd3/sd3.png, 28 NFE.
- SDXL: sampler_runs/smoke_generation/sdxl.png, 30 NFE.
- Resources: TPDM source/controller, AYS Quickstart/schedule, benchmark data/evaluator weights
  downloaded; OYS paper saved, official code not found. See RESOURCES.md.

Still pending: full adaptive SD3 Pipeline parity, real persistent Agent bridge, the
actual 10/40 manifests, baseline adaptation/reproduction, and independent evaluation.
No adaptive-strategy speed or quality claim has been measured.

Verification and setup logs: sampler_runs/setup/logs/.
