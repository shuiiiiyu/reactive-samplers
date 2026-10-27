# Metrics

Formal per-image feedback uses CLIPScore (`clip_score`) and ImageReward (`image_reward`).
Both are enabled in `conf/sd3_main.yaml`, inherited by SDXL and all three sampling methods.
HPS is no longer part of the formal scoring or feedback pipeline; historical results are retained.

CLIP uses `clip/checkpoints/ViT-B-32.pt`. Official ImageReward v1.0 uses
`image_reward/checkpoints/ImageReward.pt` and `med_config.json` from THUDM/ImageReward.
Its BLIP tokenizer also needs the cached `bert-base-uncased` tokenizer.
Install the project's metrics extra into the inference environment; it pins Transformers
4.44.2 because ImageReward 1.5 imports legacy Transformers APIs. No model training is required.

Each image is scored after saving. `results.json` and `feedback.json` retain both scores,
`scoring_seconds`, and matching prompt/seed reference scores/deltas when available.
The next Agent request includes `new_completed_images`; no quality models run inside choose().
Scoring time is separate from the existing sampling timer. ImageReward measures learned
preference and CLIPScore measures text-image alignment; neither guarantees photorealism.
Old references without ImageReward do not produce an ImageReward delta: regenerate or
rescore them with the same scorer before drawing paired conclusions.

Shared CLIP, CLIPScore and FID sources remain in clip/, clipscore/ and fid/.
