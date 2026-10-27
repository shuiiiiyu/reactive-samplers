# Retained experiment deliverables

This directory retains images, final results, input/configuration records, final
policies and compact sampling trajectories. Historical setup/smoke runs, Agent
conversations, checkpoints, temporary workspaces and intermediate program versions
were intentionally removed at the user's request.

For mini20 and COCO-10 runs:

- `images/`: generated images; `results.json`: final recorded metrics and validity.
- `config.json`, `prompts.jsonl`, `environment.json`: reproduction context.
- `final_policies/<version>/`: exported strategy sources, relative Python helpers,
  and method/hash manifest. Load using the normal Programs/frozen-policy interface.
- `final_policies/index.json`: final versions per completed sample and retained
  last-committed versions for unfinished samples.
- `trajectories/<sample>/events.jsonl`: compact step records (sigma/timestep,
  actual keep/skip or reuse actions and version), without observation dumps or
  Agent output paths. The gallery can still display sampling trajectories.
- `comparison.md` / `.csv`: retained paired summaries. Audit and validity markers
  remain as recorded; cleanup does not turn interrupted/invalid runs into successes.

Exploration can change strategy within one image. Exported final strategies alone
cannot exactly replay intermediate policy changes, whose source revisions have
been removed. Existing runs are retained deliverables, not resumable workspaces;
start a new output directory for further sampling.

`datasets/` retains prompt splits; `web/` retains web generation deliverables and
request metadata; `gallery/` retains service state. `setup/research-assets.json`
retains resource-download provenance. Models, datasets outside this directory,
installed environments, formal source and regression tests were not removed.
