"""Download only default safetensors Diffusers components; never print tokens."""
import argparse
import json
from pathlib import Path
import time


REPOS = {
    "sd3-medium": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sdxl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
}


def required_file(name):
    parts = name.split("/")
    if name == "model_index.json":
        return True
    if len(parts) < 2:
        return False
    component, filename = parts[0], parts[-1]
    if component == "scheduler" or component.startswith("tokenizer"):
        return not filename.endswith((".bin", ".onnx", ".msgpack"))
    if component not in {"transformer", "unet", "vae", "text_encoder", "text_encoder_2", "text_encoder_3"}:
        return False
    if filename == "config.json":
        return True
    if any(variant in filename for variant in (".fp16.", ".fp32.", ".non_ema.", ".bf16.")):
        return False
    return filename.endswith((".safetensors", ".safetensors.index.json"))


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("models"))
    parser.add_argument("--model", choices=[*REPOS, "both"], default="both")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "download-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
    failed = False
    for key in REPOS if args.model == "both" else [args.model]:
        repo = REPOS[key]
        status[key] = {"repo": repo, "status": "checking", "path": str((args.root / key).resolve())}
        save()
        try:
            info = HfApi().model_info(repo, files_metadata=True)
            files = sorted(f.rfilename for f in info.siblings if required_file(f.rfilename))
            if "model_index.json" not in files or not any(f.endswith(".safetensors") for f in files):
                raise RuntimeError("No complete default Diffusers safetensors layout was found")
            revision = info.sha
            destination = args.root / key
            # Check an actual weight's access, not just public metadata/config availability.
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            first_weight = next(f for f in files if f.endswith(".safetensors"))
            from huggingface_hub import get_token
            get_hf_file_metadata(hf_hub_url(repo, first_weight, revision=revision), token=get_token())
            size = sum(f.size or 0 for f in info.siblings if f.rfilename in files)
            status[key].update(status="planned" if args.plan_only else "downloading", revision=revision,
                               files=files, expected_bytes=size, started_at=time.time())
            save()
            print(json.dumps({"model": key, "status": status[key]["status"], "bytes": size}), flush=True)
            if args.plan_only:
                continue
            snapshot_download(repo, revision=revision, local_dir=destination, allow_patterns=files, max_workers=4)
            missing = [f for f in files if not (destination / f).is_file()]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} required files after download")
            status[key].update(status="complete", completed_at=time.time())
            save()
            print(json.dumps({"model": key, "status": "complete"}), flush=True)
        except Exception as exc:
            failed = True
            # The exception type is sufficient in durable metadata; no credentials/URLs are saved.
            kind = type(exc).__name__
            status[key].update(status="blocked" if "Gated" in kind else "failed", error_type=kind)
            save()
            print(json.dumps({"model": key, "status": status[key]["status"], "error_type": kind}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
