"""Readable Agent reports with separately recorded, trusted execution outcomes."""
import json
import re

from .report import write_json

HEADINGS = {
    "signals_used": "Signals used this round",
    "rule_changes": "Actions this round (rules modified or added; None if unchanged)",
    "rationale": "Why these signal conditions support these actions",
    "signals_added": "New signals added (None if none)",
}


def parse_report(text):
    pattern = re.compile(r"^(?:#+\s*)?(?:\*\*)?(" + "|".join(re.escape(h) for h in HEADINGS.values()) + r")(?:\*\*)?:\s*", re.M)
    matches = list(pattern.finditer(text))
    values = {key: None for key in HEADINGS}
    by_heading = {heading: key for key, heading in HEADINGS.items()}
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        values[by_heading[match.group(1)]] = text[match.end():end].strip()
    return values


def save_record(root, number, record):
    folder = root / "agent_decisions"
    path = folder / f"{number:05d}.json"
    write_json(path, record)
    lines = [f"# Agent decision {number}", "", f"Sample: {record['sample_id']} · Method: {record['method']}", ""]
    for key, heading in HEADINGS.items():
        lines += [f"## {heading}", record["report"].get(key) or "Not provided; see raw answer.", ""]
    lines += ["## Framework outcome", json.dumps(record.get("outcome"), indent=2), "",
              "## Actual executions", json.dumps(record.get("executions", []), indent=2), "",
              "## Raw answer", record["raw_answer"], ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def update_record(root, number, *, outcome=None, execution=None):
    path = root / "agent_decisions" / f"{number:05d}.json"
    if not path.exists():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    if outcome is not None:
        record["outcome"] = outcome
    if execution is not None:
        record["executions"].append(execution)
    save_record(root, number, record)
