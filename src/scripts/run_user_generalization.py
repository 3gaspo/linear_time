"""Seeded seen/unseen evaluation on repeated-user and sensor panels."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.pipeline.default import run_study


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    values["study"] = values["experiment"] = "user_generalization"
    run_study(values)


if __name__ == "__main__":
    main()
