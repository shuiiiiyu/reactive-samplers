"""Load complete strategy versions. Relative imports support workspace helper modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import uuid

from .recovery import policy_deadline
from .context import METHODS


def source_files(path):
    return sorted(p for p in Path(path).rglob("*.py") if "scratch" not in p.relative_to(path).parts)


def digest(path):
    h = hashlib.sha256()
    for p in source_files(path):
        h.update(p.relative_to(path).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def copy_sources(source, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    for p in source_files(source):
        dest = target / p.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)


class Programs:
    def __init__(self, workspace, versions, method="adaptive_sigma"):
        if method not in METHODS:
            raise ValueError(f"Unknown sampling method: {method}")
        self.method = method
        self.workspace, self.versions = Path(workspace), Path(versions)
        self.versions.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.reload()

    @classmethod
    def create(cls, workspace, versions, source=None, method="adaptive_sigma"):
        if method not in METHODS:
            raise ValueError(f"Unknown sampling method: {method}")
        starting = source is None
        source = Path(source) if source else Path(__file__).parent / "prompts"
        if starting and method == "grid_skip":
            source = source / "grid_skip"
        if (source / "manifest.json").exists():
            manifest = json.loads((source / "manifest.json").read_text())
            expected = manifest["sha256"]
            if manifest.get("method", "adaptive_sigma") != method:
                raise ValueError("Frozen policy belongs to a different sampling method")
            if digest(source) != expected:
                raise ValueError("Frozen strategy hash does not match its manifest")
        if starting and method == "adaptive_sigma":
            # Select the root starter only; do not copy other methods into its workspace.
            Path(workspace).mkdir(parents=True, exist_ok=False)
            for path in source.glob("*.py"):
                shutil.copyfile(path, Path(workspace) / path.name)
        else:
            copy_sources(source, workspace)
        return cls(workspace, versions, method)

    def reload(self, source=None, timeout_seconds=30):
        source = Path(source) if source is not None else self.workspace
        version = digest(source)
        if version == self.version:
            return False
        path = self.versions / version
        if not path.exists():
            copy_sources(source, path)
        self._load(path, version, timeout_seconds)
        return True

    def _load(self, path, version, timeout_seconds=30):
        if digest(path) != version:
            raise ValueError("Program version no longer matches its content hash")
        for p in source_files(path):
            compile(p.read_bytes(), str(p), "exec")
        # Fresh namespace prevents stale helper imports and Python bytecode caching.
        namespace = "_reactive_policy_" + uuid.uuid4().hex
        package = types.ModuleType(namespace)
        package.__path__ = [str(path)]
        sys.modules[namespace] = package
        modules = {}
        try:
            for name, function in (("analysis", "analyze"), ("updater", "choose")):
                spec = importlib.util.spec_from_file_location(namespace + "." + name, path / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                with policy_deadline(timeout_seconds):
                    spec.loader.exec_module(module)
                if not callable(getattr(module, function, None)):
                    raise TypeError(f"{name}.py must define {function}()")
                modules[name] = module
        except (Exception, SystemExit):
            for key in list(sys.modules):
                if key == namespace or key.startswith(namespace + "."):
                    del sys.modules[key]
            raise
        self.analysis, self.updater, self.version = modules["analysis"], modules["updater"], version

    def restore(self, version):
        path = self.versions / version
        self._load(path, version)
        self.restore_workspace()

    def use_default(self):
        with tempfile.TemporaryDirectory(dir=self.versions) as temporary:
            source = Path(temporary)
            (source / "analysis.py").write_text(
                "FEATURE_LAYERS = []\nHISTORY_LIMIT = 0\ndef analyze(ctx, memory):\n    return {}\n")
            (source / "updater.py").write_text(
                "from reactive_samplers.context import SkipAction\n"
                "def choose(observation, ctx, memory):\n    return SkipAction(False)\n"
                if self.method == "grid_skip" else
                "from reactive_samplers.context import Action\n"
                "def choose(observation, ctx, memory):\n"
                "    if ctx.remaining_nfe == 0:\n        return Action(ctx.terminal_sigma)\n"
                "    return Action(next(s for s in ctx.reference_sigmas if s < ctx.sigma))\n")
            self.reload(source)
        self.default_hash = self.version
        self.restore_workspace()

    def restore_workspace(self):
        for p in source_files(self.workspace):
            p.unlink()
        for p in source_files(self.versions / self.version):
            dest = self.workspace / p.relative_to(self.versions / self.version)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)


def freeze(source, target, method="adaptive_sigma"):
    if method not in METHODS:
        raise ValueError(f"Unknown sampling method: {method}")
    source, target = Path(source), Path(target)
    # Validate before writing an export; do not evaluate imported code as a security sandbox.
    for p in source_files(source):
        compile(p.read_bytes(), str(p), "exec")
    for name in ("analysis.py", "updater.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(name)
    copy_sources(source, target)
    (target / "manifest.json").write_text(json.dumps({"sha256": digest(target), "method": method}, indent=2) + "\n")
    return target
