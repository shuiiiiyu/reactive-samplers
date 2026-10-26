import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from reactive_samplers.gallery import Gallery, handler
from reactive_samplers.generation import normalize_request, write_atomic


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.runs = self.project / "sampler_runs"
        self.runs.mkdir()

    def test_legacy_smoke_paths_and_main_experiment_configs(self):
        smoke = self.runs / "old_smoke"
        smoke.mkdir()
        (smoke / "sdxl.png").write_bytes(b"png")
        write_atomic(smoke / "results.json", {"prompt": "red panda", "seed": 42, "images": {
            "sdxl": {"status": "complete", "image": "/old/runpod/path/sdxl.png", "steps": 30}}})
        main = self.runs / "main"
        (main / "images").mkdir(parents=True)
        (main / "images" / "p1.png").write_bytes(b"png")
        write_atomic(main / "config.json", {"guidance_scale": 7, "model_path": "models/sd3-medium",
                     "agent": {"command": "private-secret", "schedule": "every_step"}})
        write_atomic(main / "results.json", {"images": [{"id": "p1", "image": "/old/main/images/p1.png", "prompt": "test", "seed": 3, "nfe": 12}]})
        trace = main / "trajectories" / "p1" / "events.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text('{"event":"step","step":0,"sigma":1,"next_sigma":0.5,"version":"v1"}\npartial', encoding="utf-8")
        (main / "images" / "image.partial.png").write_bytes(b"incomplete")
        rows = Gallery(self.project).images()
        self.assertEqual(len(rows), 2)
        configs = {r["config"]["seed"]: r["config"] for r in rows}
        self.assertEqual(configs[42]["model"], "sdxl")
        self.assertEqual(configs[3]["guidance_scale"], 7)
        self.assertNotIn("private-secret", json.dumps(rows))
        self.assertEqual(configs[3]["agent_schedule"], {"schedule": "every_step"})
        self.assertEqual(configs[3]["sampling_trajectory"][0]["next_sigma"], 0.5)

    def test_validation_rejects_invalid_values_and_unknown_actions(self):
        base = {"model": "sd3", "prompt": "test"}
        self.assertEqual(normalize_request(base)["steps"], 28)
        for change in ({"model": []}, {"prompt": " "}, {"seed": True}, {"steps": 0},
                       {"guidance_scale": float("nan")}, {"width": 1025}, {"command": "echo unsafe"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_request({**base, **change})

    def test_experiment_groups_keep_variants_and_other_experiments_separate(self):
        for experiment, variant, method in (("mini20-20261026", "baseline_explore", "adaptive_sigma"),
                                            ("mini20-20261026", "agent_explore", "grid_skip"),
                                            ("another-experiment", "agent_explore", "adaptive_sigma")):
            run = self.runs / experiment / variant
            (run / "images").mkdir(parents=True)
            (run / "images" / "p1.png").write_bytes(b"png")
            write_atomic(run / "config.json", {"method": method})
            write_atomic(run / "results.json", {"images": [{"image": "images/p1.png", "prompt": "same", "seed": 42}]})
        other = self.runs / "another-experiment"
        write_atomic(other / "experiment.json", {"title": "仅跳过时间点对比实验"})
        rows = Gallery(self.project).images()
        main = [r for r in rows if r["experiment"]["id"] == "mini20-20261026"]
        self.assertEqual(len(main), 2)
        self.assertEqual(main[0]["experiment"]["title"], "小规模采样对比实验 · 2026-10-26")
        self.assertEqual({r["config"]["method"] for r in main}, {"adaptive_sigma", "grid_skip"})
        self.assertEqual(next(r for r in rows if r not in main)["experiment"]["title"], "仅跳过时间点对比实验")

    def test_http_submission_safe_paths_and_automatic_discovery(self):
        template = self.project / "job.sbatch"
        template.write_text("#!/bin/bash\n", encoding="utf-8")
        gallery = Gallery(self.project, template)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(gallery))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        self.addCleanup(connection.close)
        request = {"model": "sdxl", "prompt": "$(touch do-not-run); <script>test</script>", "seed": 11}
        headers = {"Content-Type": "application/json", "X-Requested-With": "reactive-samplers"}
        with patch("reactive_samplers.gallery.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "12345\n", "")) as submit:
            connection.request("POST", "/api/generate", json.dumps(request), headers)
            response = connection.getresponse()
            result = json.loads(response.read())
            self.assertEqual(response.status, 202)
            argv = submit.call_args.args[0]
            self.assertEqual(argv[0], "sbatch")
            self.assertNotIn(request["prompt"], argv)
        run = self.runs / "web" / result["id"]
        self.assertEqual(json.loads((run / "request.json").read_text())["prompt"], request["prompt"])
        (run / "new.png").write_bytes(b"real-image-bytes")
        connection.request("GET", "/api/images")
        response = connection.getresponse()
        images = json.loads(response.read())["images"]
        self.assertEqual(len(images), 1)
        connection.request("GET", images[0]["url"])
        response = connection.getresponse()
        self.assertEqual(response.read(), b"real-image-bytes")
        secret = self.project / "secret.png"
        secret.write_bytes(b"private")
        for url in ("/images/..%2Fsecret.png", "/images/%2Fetc%2Fpasswd"):
            connection.request("GET", url)
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
        connection.request("POST", "/api/generate", json.dumps(request), {**headers, "Origin": "https://other.example"})
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()


if __name__ == "__main__":
    unittest.main()
