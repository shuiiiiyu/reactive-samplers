"""Fetch official baseline sources and benchmark inputs; never execute vendored code."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "sampler_runs/setup/research-assets.json"
SOURCE_DIRS = {
    "TPDM": "baselines/tpdm/source",
    "HPSv2": "benchmarks/hps/source",
    "GenEval": "benchmarks/geneval/source",
    "mmdetection": "benchmarks/geneval/dependencies/mmdetection",
    "CLIPScore": "metrics/clipscore/source",
    "CLIP": "metrics/clip/source",
    "pytorch-fid": "metrics/fid/source",
}
LOCK = threading.Lock()
RECORDS = {}


def record(name, **fields):
    with LOCK:
        RECORDS.setdefault(name, {}).update(fields)
        temporary = STATUS.with_suffix(".tmp")
        temporary.write_text(json.dumps(RECORDS, indent=2) + "\n")
        temporary.replace(STATUS)
    print(json.dumps({"asset": name, **fields}), flush=True)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected_sha=None):
    import requests
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".part")
        for attempt in range(3):
            try:
                size = temporary.stat().st_size if temporary.exists() else 0
                with requests.get(url, stream=True, headers={"Range": f"bytes={size}-"} if size else {}, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with temporary.open("ab" if size and response.status_code == 206 else "wb") as out:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            out.write(chunk)
                temporary.replace(path)
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2)
    actual = checksum(path)
    if expected_sha and actual != expected_sha:
        raise ValueError(f"Checksum mismatch for {path.name}")
    return {"url": url, "path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def repo(name, url, branch=None):
    destination = ROOT / SOURCE_DIRS[name]
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--depth", "1"]
        if branch:
            command += ["--branch", branch]
        subprocess.run(command + [url, str(destination)], check=True)
    remote = subprocess.check_output(["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip()
    if remote != url:
        raise ValueError("Existing reference repository has a different origin")
    revision = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    return {"url": url, "path": str(destination), "revision": revision, "read_only_reference": True}


def hub(repo_id, target, patterns, repo_type="model"):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().repo_info(repo_id, repo_type=repo_type, files_metadata=True)
    files = sorted(f.rfilename for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, p) for p in patterns))
    if not files:
        raise RuntimeError(f"No matching official assets found in {repo_id}")
    destination = ROOT / target
    snapshot_download(repo_id, repo_type=repo_type, revision=info.sha, local_dir=destination,
                      allow_patterns=files, max_workers=4)
    missing = [f for f in files if not (destination / f).is_file()]
    if missing:
        raise RuntimeError(f"Missing files in {repo_id}: {missing}")
    return {"repo_id": repo_id, "repo_type": repo_type, "revision": info.sha,
            "path": str(destination), "files": files,
            "bytes": sum((destination / f).stat().st_size for f in files)}


def coco(archive_name, url, target, expected_images=None):
    folder = ROOT / "benchmarks/coco/data"
    archive = ROOT / "benchmarks/coco/archives" / archive_name
    result = download(url, archive)
    marker = archive.with_name(archive_name + ".extracted")
    if not marker.exists():
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                destination = (folder / item.filename).resolve()
                if not destination.is_relative_to(folder.resolve()):
                    raise ValueError("Unsafe ZIP member")
            bad = z.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member {bad}")
            z.extractall(folder)
        marker.write_text(result["sha256"] + "\n")
    if expected_images:
        count = len(list((folder / target).glob("*.jpg")))
        if count != expected_images:
            raise ValueError(f"Expected {expected_images} COCO reference images, found {count}")
        result["image_count"] = count
    else:
        counts = {}
        for split in ("train2017", "val2017"):
            data = json.loads((folder / "annotations" / f"captions_{split}.json").read_text())
            counts[split] = {"images": len(data["images"]), "captions": len(data["annotations"])}
        result["counts"] = counts
    result["extracted_to"] = str(folder / target)
    return result


def ays():
    folder = ROOT / "baselines/ays"
    result = download("https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html", folder / "official_quickstart.html")
    data = {"source": result["url"], "note": "Published schedule values; no solver adaptation applied.",
            "sdxl": {"noise_levels": [14.615,6.315,3.771,2.181,1.342,.862,.555,.380,.234,.113,.029],
                     "timestep_indices": [999,845,730,587,443,310,193,116,53,13,0]}}
    (folder / "published_schedules.json").write_text(json.dumps(data, indent=2) + "\n")
    result["implementation_scope"] = "Official quickstart source and published SDXL schedule, not a separately released optimizer repository"
    return result


def oys():
    folder = ROOT / "baselines/oys"
    result = download("https://arxiv.org/html/2608.18040v1", folder / "paper.html")
    download("https://arxiv.org/pdf/2608.18040v1", folder / "paper.pdf")
    (folder / "IMPLEMENTATION_STATUS.md").write_text(
        "# OYS\n\nOfficial implementation not located in the supplied paper or public project search.\n"
        "Paper archived; no third-party code is labeled as official and no reproduction is implemented yet.\n"
        "Source: https://arxiv.org/abs/2608.18040\n")
    return {**result, "status": "official_code_not_found", "paper_downloaded": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Fetch selected asset keys only; preserve other records")
    args = parser.parse_args()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    if STATUS.exists():
        RECORDS.update(json.loads(STATUS.read_text()))
    jobs = {
        "TPDM_source": lambda: repo("TPDM", "https://github.com/maple-research-lab/TPDM.git"),
        "HPSv2_source": lambda: repo("HPSv2", "https://github.com/tgxs002/HPSv2.git"),
        "GenEval_source": lambda: repo("GenEval", "https://github.com/djghosh13/geneval.git"),
        "mmdetection_source": lambda: repo("mmdetection", "https://github.com/open-mmlab/mmdetection.git", "2.x"),
        "CLIPScore_source": lambda: repo("CLIPScore", "https://github.com/jmhessel/clipscore.git"),
        "CLIP_source": lambda: repo("CLIP", "https://github.com/openai/CLIP.git"),
        "FID_source": lambda: repo("pytorch-fid", "https://github.com/mseitzer/pytorch-fid.git"),
        "AYS": ays,
        "OYS": oys,
        "TPDM_SD3_weights": lambda: hub("MAPLE-WestLake-AIGC/TPDM", "baselines/tpdm/checkpoints", ["sd3/*"]),
        "HPS_benchmark_and_HPD_metadata": lambda: hub("zhwang/HPDv2", "benchmarks/hps/data", ["benchmark/*.json", "train.json", "test.json", "README.md"], "dataset"),
        "HPD_train_metadata": lambda: hub("ymhao/HPDv2", "benchmarks/hps/training_data", ["train.json", "test.json", "README.md", "HPDv2.py"], "dataset"),
        "HPSv2_1_weights": lambda: hub("xswu/HPSv2", "benchmarks/hps/checkpoints/hpsv2", ["HPS_v2.1_compressed.pt"]),
        "HPS_base_CLIP": lambda: hub("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "benchmarks/hps/checkpoints/base_clip", ["open_clip_pytorch_model.bin", "open_clip_config.json"]),
        "COCO2017_annotations": lambda: coco("annotations_trainval2017.zip", "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", "annotations"),
        "COCO2017_val_images": lambda: coco("val2017.zip", "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", "val2017", 5000),
        "GenEval_detector": lambda: download("https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth", ROOT / "benchmarks/geneval/checkpoints/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"),
        "CLIP_ViT_L14": lambda: download("https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt", ROOT / "metrics/clip/checkpoints/ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"),
        "CLIP_ViT_B32": lambda: download("https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt", ROOT / "metrics/clip/checkpoints/ViT-B-32.pt", "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"),
        "FID_inception": lambda: download("https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth", ROOT / "metrics/fid/checkpoints/pt_inception-2015-12-05-6726825d.pth"),
    }
    if args.only:
        unknown = set(args.only) - jobs.keys()
        if unknown:
            raise ValueError(f"Unknown asset keys: {sorted(unknown)}")
        jobs = {name: jobs[name] for name in args.only}
    failed = False
    def execute(name, job):
        record(name, status="downloading")
        try:
            result = job()
            record(name, **{**result, "status": result.get("status", "complete")})
        except Exception as exc:
            record(name, status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
            raise
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, name, job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
