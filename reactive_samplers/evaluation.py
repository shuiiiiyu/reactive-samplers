"""Quality scorers are optional and explicit; missing scores are never fabricated."""


class QualityMetrics:
    def __init__(self, names, clip_model=None):
        unknown = set(names) - {"hps_v2_1", "clip_score"}
        if unknown:
            raise ValueError(f"Unknown metrics: {sorted(unknown)}")
        self.names = names
        if "hps_v2_1" in names:
            import hpsv2
            self.hps = hpsv2
        if "clip_score" in names:
            if not clip_model:
                raise ValueError("metrics.clip_model must name the CLIP checkpoint")
            from torchmetrics.multimodal.clip_score import CLIPScore
            self.clip = CLIPScore(model_name_or_path=clip_model)

    def score(self, image_path, prompt):
        scores = {}
        if "hps_v2_1" in self.names:
            scores["hps_v2_1"] = float(self.hps.score(str(image_path), prompt, hps_version="v2.1")[0])
        if "clip_score" in self.names:
            import numpy as np
            import torch
            from PIL import Image
            with Image.open(image_path) as im:
                image = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)
            self.clip.reset()
            scores["clip_score"] = float(self.clip(image, prompt))
        return scores
