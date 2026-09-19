"""Regenerate one Linear TIME study report without fitting."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.pipeline.report import write_study_report


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    values["experiment"] = values["study"]
    write_study_report(values)


if __name__ == "__main__":
    main()
