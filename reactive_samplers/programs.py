"""Load complete strategy versions. Relative imports support workspace helper modules."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
import uuid


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
    def __init__(self, workspace, versions):
        self.workspace, self.versions = Path(workspace), Path(versions)
        self.versions.mkdir(parents=True, exist_ok=True)
        self.version = None
        self.reload()

    @classmethod
    def create(cls, workspace, versions, source=None):
        source = Path(source) if source else Path(__file__).parent / "prompts"
        if (source / "manifest.json").exists():
            expected = json.loads((source / "manifest.json").read_text())["sha256"]
            if digest(source) != expected:
                raise ValueError("Frozen strategy hash does not match its manifest")
        copy_sources(source, workspace)
        return cls(workspace, versions)

    def reload(self):
        version = digest(self.workspace)
        if version == self.version:
            return False
        path = self.versions / version
        if not path.exists():
            copy_sources(self.workspace, path)
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
                spec.loader.exec_module(module)
                if not callable(getattr(module, function, None)):
                    raise TypeError(f"{name}.py must define {function}()")
                modules[name] = module
        except Exception:
            for key in list(sys.modules):
                if key == namespace or key.startswith(namespace + "."):
                    del sys.modules[key]
            raise
        self.analysis, self.updater, self.version = modules["analysis"], modules["updater"], version
        return True

    def restore_workspace(self):
        for p in source_files(self.workspace):
            p.unlink()
        for p in source_files(self.versions / self.version):
            dest = self.workspace / p.relative_to(self.versions / self.version)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)


def freeze(source, target):
    source, target = Path(source), Path(target)
    # Validate before writing an export; do not evaluate imported code as a security sandbox.
    for p in source_files(source):
        compile(p.read_bytes(), str(p), "exec")
    for name in ("analysis.py", "updater.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(name)
    copy_sources(source, target)
    (target / "manifest.json").write_text(json.dumps({"sha256": digest(target)}, indent=2) + "\n")
    return target
