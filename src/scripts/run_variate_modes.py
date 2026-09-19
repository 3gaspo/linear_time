"""Compare independent, jointly informed, and shared panel ridge models."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.pipeline.default import run_study


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    values["study"] = values["experiment"] = "variate_modes"
    run_study(values)


if __name__ == "__main__":
    main()
