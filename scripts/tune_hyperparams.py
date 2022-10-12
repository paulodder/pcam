import wandb
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
from decouple import config
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn import functional as F

# from src.resnet import ResNet, ResBlock
# from src.GCNN import GResNet18, GResnet50, GResNet34
from src.densenet import fA_P4DenseNet, fA_P4MDenseNet, P4MDenseNet, P4DenseNet
from src.dataset import get_dataloader


NET_STR2INIT_FUNC = {
    "fA_P4DenseNet": fA_P4DenseNet,
    "fA_P4MDenseNet": fA_P4MDenseNet,
    "P4MDenseNet": P4MDenseNet,
    "P4DenseNet": P4DenseNet,
}

MODEL_NAME2NUM_CHANNELS = {
    "fA_P4DenseNet": 13,
    "fA_P4MDenseNet": 9,
    "P4MDenseNet": 9,
    "P4DenseNet": 13,
}


def reduce_plat(optimizer):
    return {
        "scheduler": ReduceLROnPlateau(optimizer, "min"),
        "monitor": "val_loss",
        "frequency": 1,
    }


def reduce_step(optimizer, step_size):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


SCHED_STR2INIT_FUNC = {"reduce_plat": reduce_plat, "reduce_step": reduce_step}


class PCAMPredictor(pl.LightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config,
        run_i=0,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.run_i = run_i

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
        # lr_scheduler = SCHED_STR2INIT_FUNC[self.optimizer_config["scheduler"]](
        #     optimizer, **self.optimizer_config["scheduler_params"]
        # )
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        # wandb.log({"loss": loss, "acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        return {
            f"loss": loss,
            f"acc": acc,
        }

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
                    f"run{self.run_i}_{dataloader_name}_acc": acc,
                    f"run{self.run_i}_{dataloader_name}_loss": loss,
                }
            )
            if dataloader_name == "validation":
                self.log("val_loss", loss)
                self.val_losses.append(loss)

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
        # wandb.log({"test_acc": acc, "test_loss": loss})
        # self.log("test_acc", acc)

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


def evaluate_model():
    global wandb_config
    # Create wandb config
    wandb_config = {
        "model_signature": None,  # Set dynamically
        "dataset_config": {
            "batch_size": 64,
            "mask_types": ["otsu_split"],  # Sorted after
            "preprocess": "stain_normalize",
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": 0.0001,
            "lr": None,  # Set dynamically
        },
        "model_config": {
            "model_type": "P4DenseNet",
            "n_channels": None,  # Set dynamically
            "in_channels": None,  # Set dynamically
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "validation" if DEBUG else "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 2 if DEBUG else 75,
        "ngpus": 1,
    }
    # Sort mask types
    wandb_config["dataset_config"]["mask_types"] = sorted(
        wandb_config["dataset_config"]["mask_types"]
    )

    # Set number of channels for model
    wandb_config["model_config"]["n_channels"] = MODEL_NAME2NUM_CHANNELS[
        wandb_config["model_config"]["model_type"]
    ]

    # Get number of input channels
    ds_conf = wandb_config["dataset_config"]
    NUM_IN_CHANNELS = (
        3 + len(ds_conf["mask_types"]) + int(ds_conf["binary_mask"] is True)
    )
    wandb_config["model_config"]["in_channels"] = NUM_IN_CHANNELS

    # Get dataloaders
    split2loader = {
        split: get_dataloader(split, **ds_conf)
        for split in ["test", "validation", "train"]
    }

    min_val_losses = []

    for i in range(AVERAGE_OVER):
        model = PCAMPredictor(
            wandb_config["model_config"],
            wandb_config["optimizer_config"],
            run_i=i,
        )

        wandb_config["model_signature"] = str(model).split("\n")

        if i == 0:
            # Initialize wandb
            wandb.init(config=wandb_config)

            # Set learning rate according to sweep parameters.
            wandb_config["optimizer_config"]["lr"] = wandb.config.lr
            model.optimizer_config["lr"] = wandb_config["optimizer_config"][
                "lr"
            ]
            run_name = wandb.run.name

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=config("MODEL_DIR"),
            filename=f"{run_name}"
            + f"-lr={wandb_config['optimizer_config']['lr']:.3f}"
            + "-{epoch:02d}-{val_loss:.2f}",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        trainer = pl.Trainer(
            gpus=wandb_config["ngpus"] if torch.cuda.is_available() else 0,
            max_epochs=wandb_config["max_epochs"],
            num_sanity_val_steps=1 if DEBUG else 0,
            callbacks=[checkpoint_callback, lr_monitor],
        )
        trainer.fit(
            model,
            train_dataloaders=split2loader[wandb_config.get("train_on")],
            val_dataloaders=[
                split2loader[x] for x in wandb_config.get("validate_on")
            ],
        )
        min_val_losses.append(min(model.val_losses))

    wandb.log(
        {
            f"validation_loss_min_avg": np.mean(min_val_losses),
        }
    )


if __name__ == "__main__":
    DEBUG = True
    AVERAGE_OVER = 1 if DEBUG else 3

    sweep_configuration = {
        "method": "grid",  # options: [bayes, grid, random]
        "name": "lr_sweep" + "_DEBUG" if DEBUG else "",
        "metric": {"goal": "minimize", "name": "validation_loss_min_avg"},
        "parameters": {
            "lr": {
                "values": [0.0001, 0.001]
                if DEBUG
                else [0.0001, 0.001, 0.0025, 0.005]
            },
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="pcam", entity="pcam"
    )
    # Start sweep job.
    n_sweeps = len(sweep_configuration["parameters"]["lr"])
    wandb.agent(sweep_id, function=evaluate_model, count=n_sweeps)

    # evaluate_model()
    # TODO: implement step scheduler and vary steps
