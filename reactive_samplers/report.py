import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def event(self, kind, **fields):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": kind, **fields}, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(rows):
    if not rows:
        raise ValueError("No results to summarize")
    fields = ("nfe", "denoiser_items", "sampling_seconds", "end_to_end_seconds", "agent_seconds")
    scores = sorted({k for row in rows for k in row.get("scores", {})})
    quality = {f"mean_{k}": sum(r["scores"][k] for r in rows) / len(rows) for k in scores}
    return {"count": len(rows), **{f"mean_{k}": sum(r[k] for r in rows) / len(rows) for k in fields}, **quality}
