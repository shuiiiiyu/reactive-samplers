"""Create disjoint 5+5 exploration and 5+5 frozen-evaluation prompt sets."""

import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "sampler_runs/datasets/coco_hpd_20_v1"
RNG = random.Random(20261026)


def unique(rows, key):
    seen, result = set(), []
    for row in rows:
        prompt = row[key].strip()
        if prompt and prompt not in seen:
            seen.add(prompt)
            result.append(row)
    return result


def main():
    coco = json.loads((ROOT / "benchmarks/coco/data/annotations/captions_train2017.json").read_text())
    hpd = json.loads((ROOT / "benchmarks/hps/training_data/train.json").read_text())
    coco_rows = RNG.sample(unique(coco["annotations"], "caption"), 10)
    hpd_rows = RNG.sample(unique(hpd, "prompt"), 10)
    explore, holdout = [], []
    for source, rows, field in (("coco_train2017", coco_rows, "caption"),
                                ("hpd_v2_train", hpd_rows, "prompt")):
        for index, row in enumerate(rows):
            record = {"id": f"{source}_{index:02d}", "source": source,
                      "prompt": row[field].strip(), "seed": RNG.randrange(2**32)}
            if source == "coco_train2017":
                record["image_id"] = row["image_id"]
                record["annotation_id"] = row["id"]
            (explore if index < 5 else holdout).append(record)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("explore", explore), ("holdout", holdout)):
        (OUTPUT / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "selection_seed": 20261026, "explore_count": len(explore), "holdout_count": len(holdout),
        "sources": ["COCO 2017 train captions", "HPD v2 training prompts"],
        "rule": "Unique prompts per source; random sample of ten, first five explore, last five holdout."
    }, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
