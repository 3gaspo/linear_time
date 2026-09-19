"""Fit/evaluate the default TIME experiment on a prepared execution host."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.pipeline.default import run_study


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    values["study"] = values["experiment"] = "default"
    run_study(values)


if __name__ == "__main__":
    main()
