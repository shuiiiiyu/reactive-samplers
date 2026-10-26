"""Paired quality/speed report for the ten untouched holdout prompts."""

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1] / "sampler_runs/mini20-20261026"


def main():
    default_config = json.loads((ROOT / "baseline_holdout/config.json").read_text())
    frozen_config = json.loads((ROOT / "frozen_holdout/config.json").read_text())
    for key in ("model_path", "dtype", "height", "width", "guidance_scale", "max_sequence_length", "reference_steps", "metrics"):
        assert default_config[key] == frozen_config[key], f"Unpaired configuration: {key}"
    baseline = json.loads((ROOT / "baseline_holdout/results.json").read_text())
    frozen = json.loads((ROOT / "frozen_holdout/results.json").read_text())
    reference = {row["id"]: row for row in baseline["images"]}
    assert set(reference) == {row["id"] for row in frozen["images"]}
    paired = []
    for row in frozen["images"]:
        original = reference[row["id"]]
        assert (row["prompt"], row["seed"]) == (original["prompt"], original["seed"])
        assert row["agent_calls"] == 0
        assert row.get("valid_for_benchmark", True) and original.get("valid_for_benchmark", True), \
            "Recovered/default-fallback samples require a separate report; do not silently pool them"
        paired.append({"id": row["id"], "source": row["source"],
                       "baseline": original, "frozen": row})
    summaries = {}
    for source in ["all", *sorted({row["source"] for row in paired})]:
        rows = [row for row in paired if source == "all" or row["source"] == source]
        before, after = [row["baseline"] for row in rows], [row["frozen"] for row in rows]
        summary = {"count": len(rows)}
        for field in ("nfe", "steps", "sampling_seconds", "end_to_end_seconds"):
            summary[f"baseline_{field}"] = mean(row[field] for row in before)
            summary[f"frozen_{field}"] = mean(row[field] for row in after)
        for metric in ("hps_v2_1", "clip_score"):
            summary[f"baseline_{metric}"] = mean(row["scores"][metric] for row in before)
            summary[f"frozen_{metric}"] = mean(row["scores"][metric] for row in after)
            summary[f"delta_{metric}"] = summary[f"frozen_{metric}"] - summary[f"baseline_{metric}"]
        summary["sampling_speedup"] = summary["baseline_sampling_seconds"] / summary["frozen_sampling_seconds"]
        summary["end_to_end_speedup"] = summary["baseline_end_to_end_seconds"] / summary["frozen_end_to_end_seconds"]
        summaries[source] = summary
    report = {"summary": summaries, "paired_images": paired,
              "note": "Small disjoint holdout; no FID. Latencies exclude scoring and include per-image prompt encoding/decoding in end_to_end_seconds."}
    (ROOT / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Mini20 frozen-policy evaluation", "", report["note"], "",
             "| Source | N | NFE default / frozen | HPS default / frozen | CLIPScore default / frozen | Sampling speedup | End-to-end speedup |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for source, row in summaries.items():
        lines.append(f"| {source} | {row['count']} | {row['baseline_nfe']:.2f} / {row['frozen_nfe']:.2f} | "
                     f"{row['baseline_hps_v2_1']:.5f} / {row['frozen_hps_v2_1']:.5f} | "
                     f"{row['baseline_clip_score']:.3f} / {row['frozen_clip_score']:.3f} | "
                     f"{row['sampling_speedup']:.3f}x | {row['end_to_end_speedup']:.3f}x |")
    (ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
