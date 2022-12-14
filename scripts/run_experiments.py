from argparse import ArgumentParser
from src.managers import ExperimentManager
from src.experiment_constants import (
    get_config,
    EXPERIMENT_NAME2REPEATS,
    EXPERIMENT_NAME2CONDITIONS,
    LAYERS,
)
import pandas as pd
import wandb

from src.run_config import run_config


# data.append(
#     {
#         "experiment_name": wconfig["experiment_name"],
#         "mask_types": tuple(sorted(wconfig["mask_types"])),
#     }
# )


def get_mock_args(overwrite_kwargs=dict()):
    "for dev purposes"

    class Args(object):
        pass

    mock_args = Args()
    for name, default_value in overwrite_kwargs.items():
        setattr(mock_args, name, overwrite_kwargs.get(name, default_value))
    return mock_args


def get_parser():
    parser = ArgumentParser()
    # dataloader arguments
    parser.add_argument("--experiment_name", default="LAYERS", type=str)
    return parser


def parse_args():
    return get_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp_manager = ExperimentManager(args.experiment_name)
    # necessary_run_configs = exp_manager.get_necessary_runs()
    while True:
        import random
        necessary_run_configs = (exp_manager.get_necessary_runs())
        random.shuffle(necessary_run_configs)
        if len(necessary_run_configs) == 0:
            break
        print("gonna run")
        print(necessary_run_configs[0])
        run_config(necessary_run_configs[0])
