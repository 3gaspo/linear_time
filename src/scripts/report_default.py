"""Regenerate a complete default report without refitting."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.pipeline.report import write_study_report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    values["study"] = values["experiment"] = "default"
    write_study_report(values)


if __name__ == "__main__":
    main()
