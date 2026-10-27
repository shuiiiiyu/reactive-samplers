"""Configuration defaults and generated JSON agree; no model or Agent is invoked."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import prepare_coco10_grid


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_defaults_and_step_override_resolve_for_all_entry_configs(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
            for name, model, method, steps in (
                ("sd3_main", "sd3", "grid_skip", 28),
                ("sd3_grid_skip", "sd3", "grid_skip", 28),
                ("sdxl_main", "sdxl", "grid_skip", 30),
            ):
                cfg = OmegaConf.to_container(compose(config_name=name), resolve=True, throw_on_missing=True)
                self.assertEqual((cfg["model"], cfg["method"], cfg["reference_steps"]), (model, method, steps))
                self.assertEqual(cfg["max_nfe"], steps)
                self.assertEqual(cfg["metrics"]["names"], ["clip_score", "image_reward"])
                self.assertNotIn("hps_checkpoint", cfg["metrics"])
                changed = OmegaConf.to_container(compose(config_name=name, overrides=["reference_steps=40"]),
                                                resolve=True, throw_on_missing=True)
                self.assertEqual(changed["max_nfe"], 40)

    def test_coco_generator_reads_yaml_steps_and_preserves_existing_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "conf", root / "conf")
            config = root / "conf/sd3_main.yaml"
            config.write_text(config.read_text().replace("reference_steps: 28", "reference_steps: 37"))
            annotations = root / "benchmarks/coco/data/annotations/captions_train2017.json"
            annotations.parent.mkdir(parents=True)
            annotations.write_text(json.dumps({"annotations": [
                {"id": i, "image_id": i, "caption": f"fixture caption {i}"} for i in range(12)]}))
            argv = ["prepare_coco10_grid.py", "--name", "fixture", "--runtime-root", "/runtime/project"]
            with patch.object(prepare_coco10_grid, "ROOT", root), patch.object(sys, "argv", argv):
                prepare_coco10_grid.main()
                output = root / "sampler_runs/fixture"
                for arm in ("default", "skip"):
                    cfg = json.loads((output / f"sd3_{arm}.json").read_text())
                    self.assertEqual((cfg["reference_steps"], cfg["max_nfe"]), (37, 37))
                self.assertEqual(json.loads((output / "sdxl_default.json").read_text())["reference_steps"], 30)
                before = (output / "sd3_default.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    prepare_coco10_grid.main()
                self.assertEqual((output / "sd3_default.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
