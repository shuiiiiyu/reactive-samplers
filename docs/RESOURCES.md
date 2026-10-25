# Downloaded resources

All paths below are relative to this project root. Assets are grouped by their purpose.

| Resource | Source/code | Data or checkpoint |
| --- | --- | --- |
| SD3-Medium | Official Diffusers pipeline | models/sd3-medium/ |
| SDXL Base 1.0 | Official Diffusers pipeline | models/sdxl-base-1.0/ |
| TPDM | baselines/tpdm/source/ | baselines/tpdm/checkpoints/sd3/model.safetensors |
| AYS | baselines/ays/official_quickstart.html | baselines/ays/published_schedules.json |
| OYS | baselines/oys/paper.html and paper.pdf | Official code not located; see IMPLEMENTATION_STATUS.md |
| COCO 2017 | Official annotations | benchmarks/coco/data/annotations/ and data/val2017/ |
| HPS / HPD | benchmarks/hps/source/ | data/benchmark/, training_data/, checkpoints/ |
| GenEval | benchmarks/geneval/source/ | source/prompts/evaluation_metadata.jsonl and checkpoints/ |
| GenEval detector dependency | benchmarks/geneval/dependencies/mmdetection/ | Configuration retained in that upstream repo |
| CLIP | metrics/clip/source/ | metrics/clip/checkpoints/ |
| CLIPScore | metrics/clipscore/source/ | Uses the corresponding CLIP weights |
| FID | metrics/fid/source/ | metrics/fid/checkpoints/ |

COCO contains 118,287 training image records / 591,753 training captions and 5,000
validation image records / 25,014 validation captions; the 5,000 validation reference
JPEGs are present. Training JPEGs are unnecessary for text-only prompt exploration
and were not downloaded. Source ZIPs are kept in benchmarks/coco/archives/.

HPS Benchmark contains 800 prompts in each of anime, concept-art, paintings and photo
(3,200 total). HPD train/test annotation metadata is in benchmarks/hps/training_data/;
HPD's large image corpus is not needed for our frozen-generator prompt exploration.
GenEval's downloaded official prompt list contains 553 records; final evaluation uses
four images per prompt. GenEval never supplies exploration feedback.

HPSv2.1, its base CLIP, GenEval Mask2Former, OpenAI CLIP ViT-B/32 and ViT-L/14, and FID
Inception weights are downloaded. Downloaded sources/weights do not imply their
reproduction environments or benchmark evaluation have been run.

Exact source URLs, Git commits, HF revisions, selected files and checksums are recorded
in sampler_runs/setup/research-assets.json. Base model download records are in
models/download-status.json. One-time old-to-new path mappings are retained in
sampler_runs/setup/layout-migration.json; old logs remain historical records.
