from src.experiment_constants import (
    get_config,
    EXPERIMENT_NAME2REPEATS,
    EXPERIMENT_NAME2CONDITIONS,
    LAYERS,
)
import pandas as pd
import wandb

WANDB_PATH = "pcam/pcam"


class ExperimentManager:
    def __init__(self, experiment_name):
        self.api = wandb.Api()
        self.experiment_name = experiment_name

    def _get_runs(self):
        for run in self.api.runs(WANDB_PATH):
            if run.config.get("experiment_name") == self.experiment_name:
                if run.state in ("running", "finished"):
                    yield run

    def register_experiments_run(self):
        data = []
        for run in self._get_runs():
            wconfig = run.config
            data.append(
                {
                    "experiment_name": wconfig.get("experiment_name"),
                    "mask_types": sorted(wconfig.get("mask_types")),
                }
            )
        regdf = pd.DataFrame(data, columns=["experiment_name", "mask_types"])
        regdf = (
            regdf.groupby(["experiment_name", "mask_types"])
            .size()
            .rename("nof_runs")
            .reset_index()
        )
        regdf = regdf.rename(lambda s: f"run_{s}", axis=1)
        return regdf

    def register_experiments_needed(self):
        data = []
        nof_repeats = EXPERIMENT_NAME2REPEATS[self.experiment_name]
        for cond in EXPERIMENT_NAME2CONDITIONS[self.experiment_name]:
            data.append(
                {
                    "param2val": cond,
                    "nof_repeats": nof_repeats,
                    **cond,
                }
            )
        registry_df = pd.DataFrame(data)
        registry_df["nof_runs"] = nof_repeats
        registry_df = registry_df.rename(lambda s: f"needed_{s}", axis=1)
        return registry_df

    def get_necessary_runs(self):
        regdf_run = self.register_experiments_run()
        regdf_needed = self.register_experiments_needed()
        regdf = regdf_needed.merge(
            regdf_run,
            left_on=["needed_mask_types"],
            right_on=["run_mask_types"],
            how="left",
        )
        regdf["run_nof_runs"] = regdf["run_nof_runs"].fillna(0)
        regdf["runs_to_do"] = regdf["needed_nof_runs"] - regdf["run_nof_runs"]
        necessary_runs = []
        for _, row in regdf.iterrows():
            regdf.iloc[0].needed_param2val
            if row.runs_to_do > 0:
                necessary_runs.append(
                    get_config(self.experiment_name, row.needed_param2val)
                )  # you only need to append this one once, cause it'll
                # re-decide what to run upon each new run it starts, no need
                # to keep track of cardinality.
        return necessary_runs
