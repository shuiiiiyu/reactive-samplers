"""Ask the RunPod mailbox bridge to inspect the current paused Trento state."""
import json
from pathlib import Path
import time


def main():
    workspace = Path.cwd().resolve()
    folder = workspace / ".agent_tools"
    folder.mkdir(exist_ok=True)
    sequence = folder / "inspect-sequence"
    number = int(sequence.read_text()) + 1 if sequence.exists() else 1
    sequence.write_text(str(number))
    request = folder / f"inspect-{number:05d}.request.json"
    response = folder / f"inspect-{number:05d}.response.json"
    temporary = request.with_suffix(request.suffix + ".tmp")
    temporary.write_text(json.dumps({"workspace": str(workspace)}) + "\n", encoding="utf-8")
    temporary.replace(request)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if response.is_file():
            result = json.loads(response.read_text(encoding="utf-8"))
            if result.get("status") != "ok":
                raise RuntimeError(result.get("error", "analysis inspection failed"))
            print(json.dumps(result["observation"], ensure_ascii=False, indent=2))
            return
        time.sleep(1)
    raise TimeoutError("Timed out waiting for the Trento inspection result")


if __name__ == "__main__":
    main()
