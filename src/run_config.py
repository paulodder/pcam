from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from decouple import config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

from src.dataset import get_dataloader
from src.resnet import ResNet, ResBlock

# from src.GCNN import GResNet18, GResnet50, GResNet34
from src.densenet import fA_P4DenseNet, fA_P4MDenseNet, P4MDenseNet, P4DenseNet

# from src.pcam_predictor import PCAMPredictor
from pytorch_lightning.callbacks import LearningRateMonitor

NET_STR2INIT_FUNC = {
    "fA_P4DenseNet": fA_P4DenseNet,
    "fA_P4MDenseNet": fA_P4MDenseNet,
    "P4MDenseNet": P4MDenseNet,
    "P4DenseNet": P4DenseNet,
    # "GResNet18": GResNet18,
}


TEST_CONFS = [
    {
        "dataset_config": {
            "batch_size": 64,
            "preprocess": None,
            "binary_mask": True,
            "mask_types": ["binary_mask", "otsu_split"],
        },
        "optimizer_config": {"weight_decay": 0.0001, "lr": 0.01},
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
        "experiment_name": "TEST",
    },
    {
        "dataset_config": {
            "batch_size": 64,
            "preprocess": None,
            "binary_mask": True,
            "mask_types": ["binary_mask"],
        },
        "optimizer_config": {"weight_decay": 0.0001, "lr": 0.01},
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
        "experiment_name": "TEST",
    },
]


class PCAMPredictor(pl.LightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        model_func = NET_STR2INIT_FUNC[model_config["model_type"]]

        if "GRes" in model_config["model_type"]:
            self.model = NET_STR2INIT_FUNC[model_config["model_type"]](
                model_config["in_channels"], model_config["dropout_p"]
            )
        else:
            self.model = NET_STR2INIT_FUNC[model_config["model_type"]](
                model_config["in_channels"],
                model_config["num_blocks"],
                model_config["n_channels"],
            )
        self.loss_module = nn.BCEWithLogitsLoss()
        self.val_losses = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
        )
        return {
            "optimizer": optimizer,
        }

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        wandb.log({"loss": loss, "acc": acc})
        print(batch.shape)
        return loss
        # return {"loss": loss, "training_acc": acc}

    def validation_step(self, batch, batch_idx, dataloader_i):
        # breakpoint()
        loss, acc = self.forward(batch, mode="val")
        return {
            f"loss": loss,
            f"acc": acc,
        }

    # def validation_step(self, batch, batch_idx):
    #     # breakpoint()
    #     loss, acc = self.forward(batch, mode="val")
    #     return {
    #         f"loss": loss,
    #         f"acc": acc,
    #     }

    def validation_epoch_end(self, outputs):
        if type(outputs[0]) == dict:
            outputs = [outputs]

        for dataloader_name, output in zip(
            wandb_config["validate_on"], outputs
        ):
            acc = np.mean([tmp["acc"] for tmp in output])
            loss = np.mean([tmp["loss"].cpu() for tmp in output])
            wandb.log(
                {
                    f"{dataloader_name}_acc": acc,
                    f"{dataloader_name}_loss": loss,
                }
            )
            if dataloader_name == "validation":
                self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")

        return {
            "loss": loss,
            "acc": acc,
        }

    def test_epoch_end(self, outputs):
        acc = np.mean([tmp["acc"] for tmp in outputs])
        loss = np.mean([tmp["loss"].cpu() for tmp in outputs])
        # print(len(outputs), acc)
        wandb.log({"test_acc": acc, "test_loss": loss})

    def forward(self, data, mode="train"):
        x, y = data
        outp = self.model(x)
        y_pred_proba = F.softmax(outp)
        loss = self.loss_module(y_pred_proba, y)
        y_pred_proba = y_pred_proba.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        acc = sum(np.argmax(y_pred_proba, 1) == np.argmax(y, 1)) / len(
            y_pred_proba
        )
        return loss, acc


def run_given_config(wandb_config):
    if "binary_mask" in wandb_config["dataset_config"]["mask_types"]:
        wandb_config["dataset_config"]["binary_mask"] = True
        wandb_config["dataset_config"]["mask_types"] = [
            x
            for x in wandb_config["dataset_config"]["mask_types"]
            if x != "binary_mask"
        ]
    else:
        wandb_config["dataset_config"]["binary_mask"] = False
    wandb_config["dataset_config"]["mask_types"] = sorted(
        wandb_config["dataset_config"]["mask_types"]
    )
    ds_conf = wandb_config["dataset_config"]
    NUM_CHANNELS = (
        3 + len(ds_conf["mask_types"]) + int(ds_conf["binary_mask"] is True)
    )
    wandb_config["model_config"]["in_channels"] = NUM_CHANNELS
    print(wandb_config)
    # get dataloaders
    split2loader = {
        split: get_dataloader(split, **ds_conf)
        for split in ["test", "validation", "train"]
    }

    model = PCAMPredictor(
        wandb_config.get("model_config"),
        wandb_config["optimizer_config"],
    )
    wandb_config["model_signature"] = str(model).split("\n")
    wandb.init(
        project="pcam",
        entity="pcam",
        config=wandb_config,
    )
    run_name = wandb.run.name
    print(wandb_config)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=config("MODEL_DIR"),
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.2f}",
    )
    # lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        gpus=wandb_config["ngpus"] if torch.cuda.is_available() else 0,
        max_epochs=wandb_config["max_epochs"],
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model,
        train_dataloaders=split2loader[wandb_config.get("train_on")],
        val_dataloaders=[
            split2loader[x] for x in wandb_config.get("validate_on")
        ],
    )
    test_result = trainer.test(
        model, split2loader[wandb_config.get("test_on")], verbose=False
    )
    print(trainer.callback_metrics)


for conf in TEST_CONFS[1:]:
    run_given_config(conf)
