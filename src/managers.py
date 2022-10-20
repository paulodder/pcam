from src.experiment_constants import (
    get_config,
    EXPERIMENT_NAME2REPEATS,
    EXPERIMENT_NAME2CONDITIONS,
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
                    "param2val": wconfig.get("param2val"),
                }
            )

        regdf = pd.DataFrame(data, columns=["experiment_name", "param2val"])
        regdf = self.add_str_param2val_col(regdf)
        regdf = (
            regdf.groupby(["experiment_name", "param2val_str"])
            .size()
            .rename("nof_runs")
            .reset_index()
        )
        regdf["param2val"] = regdf["param2val_str"].apply(lambda x: eval(x))
        del regdf["param2val_str"]

        # regdf = grouped_regdf.merge(
        #     regdf[["param2val"]], left_on="param2val", right_on="param2val"
        # )

        regdf = regdf.rename(lambda s: f"run_{s}", axis=1)
        return regdf

    def register_experiments_needed(self):
        data = []
        nof_repeats = EXPERIMENT_NAME2REPEATS[self.experiment_name]
        for cond in EXPERIMENT_NAME2CONDITIONS[self.experiment_name]:
            data.append(
                {
                    **cond,
                    "param2val": cond,
                    # "param2val_str": str(cond),
                    "nof_repeats": nof_repeats,
                }
            )
        registry_df = pd.DataFrame(data)
        registry_df["nof_runs"] = nof_repeats
        registry_df = registry_df.rename(lambda s: f"needed_{s}", axis=1)
        return registry_df

    def add_str_param2val_col(self, df):
        colname = [col for col in df if "param2val" in col][0]
        df[f"{colname}_str"] = df[colname].astype(str)
        return df

    def get_necessary_runs(self):
        regdf_run = self.register_experiments_run()
        self.add_str_param2val_col(regdf_run)
        regdf_needed = self.register_experiments_needed()
        self.add_str_param2val_col(regdf_needed)
        # breakpoint()
        regdf = regdf_needed.merge(
            regdf_run,
            left_on=["needed_param2val_str"],
            right_on=["run_param2val_str"],
            how="left",
        )
        regdf["run_nof_runs"] = regdf["run_nof_runs"].fillna(0)
        regdf["runs_to_do"] = regdf["needed_nof_runs"] - regdf["run_nof_runs"]
        necessary_runs = []
        # print(regdf.runs_to_do)
        for _, row in regdf.iterrows():
            if row.runs_to_do > 0:

                conf = get_config(self.experiment_name, row.needed_param2val)
                # assert (
                #     conf["dataset_config"]["mask_types"]
                #     == row.needed_param2val["mask_types"]
                # )
                # breakpoint()
                print(f"{row.runs_to_do} repeats to do of {conf}")
                necessary_runs.append(
                    conf
                )  # you only need to append this one once, cause it'll
                # re-decide what to run upon each new run it starts, no need
                # to keep track of cardinality.

        return necessary_runs


if __name__ == "__main__":
    for experiment in EXPERIMENT_NAME2REPEATS:
        print("Experiment", experiment)
        manager = ExperimentManager(experiment)
        manager.get_necessary_runs()
        # regdf_run = manager.register_experiments_run()
        # manager.add_str_param2val_col(regdf_run)
        # regdf_needed = manager.register_experiments_needed()
        # manager.add_str_param2val_col(regdf_needed)
        # # breakpoint()
        # regdf = regdf_needed.merge(
        #     regdf_run,
        #     left_on=["needed_param2val_str"],
        #     right_on=["run_param2val_str"],
        #     how="left",
        # )
        # regdf["run_nof_runs"] = regdf["run_nof_runs"].fillna(0)
        # regdf["runs_to_do"] = regdf["needed_nof_runs"] - regdf["run_nof_runs"]

        # print(regdf)
