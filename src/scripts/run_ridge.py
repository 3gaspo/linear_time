"""Run one ridge task: PYTHONPATH=src python -m scripts.run_ridge dataset=SG_PM25/H."""

import hydra
from omegaconf import DictConfig, OmegaConf

from timebench.data.grid import load_grid
from timebench.data.time import load_time_panels
from timebench.pipeline.tasks import run_panel_task


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    setting = next(item for item in load_grid()[values["dataset"]]
        if (item.L_term, item.H_term) == (values["L_term"], values["H_term"]))
    panels = load_time_panels(values["dataset"], values["data"]["storage_path"])
    run_panel_task(values, setting, panels[0])


if __name__ == "__main__":
    main()
