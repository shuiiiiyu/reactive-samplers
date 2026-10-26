import json
import os
from pathlib import Path
import uuid


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Recorder:
    def __init__(self, root, run_root=None):
        self.root = Path(root)
        self.run_root = Path(run_root) if run_root is not None else self.root
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
    return {"count": len(rows), "recovered_count": sum(bool(r.get("recovered")) for r in rows),
            "valid_count": sum(r.get("valid_for_benchmark", True) for r in rows),
            **{f"mean_{k}": sum(r[k] for r in rows) / len(rows) for k in fields}, **quality}
