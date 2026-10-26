"""Offline quality scorers using the project's downloaded reference checkpoints."""

from pathlib import Path
import sys
import types


class QualityMetrics:
    def __init__(self, names, clip_model=None, hps_checkpoint=None):
        unknown = set(names) - {"hps_v2_1", "clip_score"}
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}")
        self.names = names
        root = Path(__file__).resolve().parents[1]
        if "hps_v2_1" in names:
            checkpoint = Path(hps_checkpoint or root / "benchmarks/hps/checkpoints/hpsv2/HPS_v2.1_compressed.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sys.path.insert(0, str(root / "benchmarks/hps/source"))
            # The upstream factory imports unused turtle.forward, which requires tkinter.
            # Keep its source untouched and provide that name only during import.
            turtle = sys.modules.get("turtle")
            if turtle is None:
                shim = types.ModuleType("turtle")
                shim.forward = None
                sys.modules["turtle"] = shim
            try:
                from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
            finally:
                if turtle is None:
                    del sys.modules["turtle"]
            import torch
            self.hps_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.hps, _, self.hps_preprocess = create_model_and_transforms(
                "ViT-H-14", pretrained=None, precision="amp", device=self.hps_device,
                light_augmentation=True, output_dict=True,
            )
            state = torch.load(checkpoint, map_location=self.hps_device, weights_only=False)
            self.hps.load_state_dict(state["state_dict"])
            self.hps.eval()
            self.hps_tokenize = get_tokenizer("ViT-H-14")
        if "clip_score" in names:
            if not clip_model:
                raise ValueError("metrics.clip_model must name the CLIP checkpoint")
            checkpoint = Path(clip_model)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sys.path.insert(0, str(root / "metrics/clip/source"))
            import clip
            import torch
            self.clip, self.clip_preprocess = clip.load(str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu")
            self.clip.eval()
            self.clip_tokenize = clip.tokenize

    def score(self, image_path, prompt):
        scores = {}
        if "hps_v2_1" in self.names:
            import torch
            from PIL import Image
            with Image.open(image_path) as im:
                image = self.hps_preprocess(im.convert("RGB")).unsqueeze(0).to(self.hps_device)
            tokens = self.hps_tokenize([prompt]).to(self.hps_device)
            with torch.inference_mode(), torch.autocast(device_type=self.hps_device, enabled=self.hps_device == "cuda"):
                output = self.hps(image, tokens)
                scores["hps_v2_1"] = float((output["image_features"] @ output["text_features"].T)[0, 0])
        if "clip_score" in self.names:
            import torch
            from PIL import Image
            with Image.open(image_path) as im:
                image = self.clip_preprocess(im.convert("RGB")).unsqueeze(0).to(next(self.clip.parameters()).device)
            tokens = self.clip_tokenize([prompt], truncate=True).to(image.device)
            with torch.inference_mode():
                image_features = self.clip.encode_image(image).float()
                text_features = self.clip.encode_text(tokens).float()
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
            scores["clip_score"] = max(100.0 * similarity, 0.0)
        return scores
