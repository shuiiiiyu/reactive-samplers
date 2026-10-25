"""One Hydra entry point for all main-experiment modes."""
import hydra
from omegaconf import OmegaConf
from reactive_samplers.cli import run_config


@hydra.main(version_base=None, config_path="../conf", config_name="sd3_main")
def main(cfg):
    run_config(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))


if __name__ == "__main__":
    main()
