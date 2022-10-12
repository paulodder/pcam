LAYERS = "LAYERS"
EXPERIMENT_NAME2WANDB_CONFIG = {
    LAYERS: {
        "dataset_config": {
            "batch_size": 64,
            # "mask_types": sorted(["pannuke-type", "otsu_split"]),
            "preprocess": None,
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": 0.001,
        },
        "model_config": {
            "model_type": "P4MDenseNet",
            "n_channels": 13,
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 75,
        "ngpus": 1,
    }
}
EXPERIMENT_NAME2REPEATS = {LAYERS: 5}
EXPERIMENT_NAME2CONDITIONS = {
    LAYERS: [
        {"mask_types": ("binary_mask", "otsu_split"), "lr": 0.01},
        {"mask_types": ("binary_mask", "otsu_split"), "lr": 0.01},
    ]
}


def get_config_for_layers(param2val):
    config = EXPERIMENT_NAME2WANDB_CONFIG[LAYERS].copy()
    config["dataset_config"]["mask_types"] = param2val["mask_types"]
    config["optimizer_config"]["lr"] = param2val["lr"]
    return config


EXPERIMENT_NAME2GET_CONFIG_FUNC = {LAYERS: get_config_for_layers}


def get_config(experiment_name, param2val):
    return EXPERIMENT_NAME2GET_CONFIG_FUNC[experiment_name](param2val)


get_config(
    LAYERS,
    {"mask_types": ("binary_mask", "pannuke-type", "otsu_split"), "lr": 3},
)
