from copy import deepcopy

LAYERS = "LAYERS_TESTING2"

PREPROCESS = "PREPROCESS"
ATTENTION = "ATTENTION"


EXPERIMENT_NAME2WANDB_CONFIG = {
    LAYERS: {
        "dataset_config": {
            "batch_size": 64,
            "preprocess": None,
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": 0.001,
        },
        "model_config": {
            "model_type": "P4DenseNet",
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 75,
        "ngpus": 1,
    },
    PREPROCESS: {
        "dataset_config": {
            "batch_size": 64,
            "mask_types": [],
            # "preprocess": None,
            "binary_mask": False,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": 0.001,
        },
        "model_config": {
            "model_type": "P4DenseNet",
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 75,
        "ngpus": 1,
    },
    ATTENTION: {
        "dataset_config": {
            "batch_size": 64,
            "mask_types": ["otsu_split", "pannuke-type"],
            "preprocess": "stain_normalize",
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": 0.001,
        },
        "model_config": {
            # "model_type": "fA_P4MDenseNet",
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 75,
        "ngpus": 1,
    },
}
EXPERIMENT_NAME2REPEATS = {LAYERS: 3, PREPROCESS: 3, ATTENTION: 3}
EXPERIMENT_NAME2CONDITIONS = {
    LAYERS: [
        {"mask_types": sorted(["binary_mask"])},
        {"mask_types": sorted([])},
        {"mask_types": sorted(["binary_mask", "otsu_split"])},
        {"mask_types": sorted(["binary_mask", "pannuke-type"])},
        {"mask_types": sorted(["binary_mask", "pannuke-type", "otsu_split"])},
    ],
    PREPROCESS: [{"preprocess": "stain_normalize"}, {"preprocess": None}],
    ATTENTION: [
        {"model_type": "fA_P4MDenseNet"},
        {"model_type": "P4MDenseNet"},
    ],
}


def get_config_for_layers(param2val):

    config = deepcopy(EXPERIMENT_NAME2WANDB_CONFIG[LAYERS])
    config["experiment_name"] = LAYERS
    config["dataset_config"]["mask_types"] = param2val["mask_types"]
    config["param2val"] = param2val
    return config


def get_config_for_attention(param2val):
    config = deepcopy(EXPERIMENT_NAME2WANDB_CONFIG[ATTENTION])
    config["experiment_name"] = ATTENTION
    config["model_config"]["model_type"] = param2val["model_type"]
    config["param2val"] = param2val

    return config


def get_config_for_preprocess(param2val):
    config = deepcopy(EXPERIMENT_NAME2WANDB_CONFIG[PREPROCESS])
    config["experiment_name"] = PREPROCESS
    config["dataset_config"]["preprocess"] = param2val["preprocess"]
    config["param2val"] = param2val
    return config


EXPERIMENT_NAME2GET_CONFIG_FUNC = {
    LAYERS: get_config_for_layers,
    PREPROCESS: get_config_for_preprocess,
    ATTENTION: get_config_for_attention,
}


def get_config(experiment_name, param2val):
    return EXPERIMENT_NAME2GET_CONFIG_FUNC[experiment_name](param2val)


# get_config(
#     LAYERS,
#     {"mask_types": ("binary_mask", "pannuke-type", "otsu_split"), "lr": 3},
# )
