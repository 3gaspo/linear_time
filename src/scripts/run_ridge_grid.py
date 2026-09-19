"""Run a dataset's nine L-H settings with the same user split."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.data.grid import load_grid
from timebench.data.time import load_time_panels
from timebench.pipeline.tasks import run_panel_task


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    panels = load_time_panels(values["dataset"], values["data"]["storage_path"])
    for setting in load_grid()[values["dataset"]]:
        for panel in panels:
            run_panel_task(values, setting, panel)


if __name__ == "__main__":
    main()
