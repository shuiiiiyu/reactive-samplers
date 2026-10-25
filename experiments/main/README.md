# Main experiment manifests

Prepare `explore.jsonl` with 10 real COCO Train / HPD training prompts and `evaluate.jsonl`
with 40 disjoint real held-out prompts. Every row must have id, prompt and seed; preserve
dataset source/record provenance in additional fields. Actual manifests are not supplied yet.
Use fixed initial noise seeds for paired comparisons. Do not optimize using evaluation feedback.

Run the every-step and fixed-checkpoint exploration variants in independent output dirs.
Each variant exports its own frozen policy, then evaluates without an Agent.
Default/reference generation uses the same model, data, seed, CFG and sampler settings.
