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
    # "GResNet18": GResNet18,
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
        lr_scheduler = SCHED_STR2INIT_FUNC[self.optimizer_config["scheduler"]](
            optimizer, **self.optimizer_config["scheduler_params"]
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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

        for dataloader_name, output in zip(run_config["validate_on"], outputs):
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
        # wandb.log({"test_acc": acc, "test_loss": loss})
        # self.log("test_acc", acc)

    def forward(self, data, mode="train"):
        x, y = data
        # breakpoint()
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
    wandb.init()
    global run_config
    run_config = {
        "dataset_config": {
            "batch_size": 64,
            "mask_type": None,
            "binary_mask": True,
        },
        "optimizer_config": {
            "weight_decay": wandb.config.weight_decay,
            "lr": wandb.config.lr,
            "scheduler": "reduce_step",
            "scheduler_params": {"step_size": wandb.config.sched_step_size},
        },
        # "optimizer_config": {
        #     "weight_decay": 0.001,
        #     "lr": 0.001,
        #     "scheduler": "reduce_step",
        #     "scheduler_params": {"step_size": 5},
        # },
        "model_config": {
            "model_type": "fA_P4DenseNet",
            "n_channels": 9,
            "dropout_p": 0.5,
            "num_blocks": 5,
        },
        "train_on": "train",
        "validate_on": ["validation"],
        "test_on": "test",
        "max_epochs": 50,
        "ngpus": 1,
    }
    ds_conf = run_config["dataset_config"]
    NUM_CHANNELS = (
        3
        + int(ds_conf["mask_type"] is not None)
        + int(ds_conf["binary_mask"] is True)
    )
    run_config["model_config"]["in_channels"] = NUM_CHANNELS

    # get dataloaders
    split2loader = {
        split: get_dataloader(split, **ds_conf)
        for split in ["test", "validation", "train"]
    }

    accs = []
    losses = []

    print(f"Optimizing parameters")
    model = PCAMPredictor(
        run_config["model_config"],
        run_config["optimizer_config"],
    )

    run_config["model_signature"] = str(model).split("\n")
    run_name = wandb.run.name

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=config("MODEL_DIR"),
        filename=f"{run_name}"
        + f"-lr={run_config['optimizer_config']['lr']:.3f}"
        + f"-wd={run_config['optimizer_config']['weight_decay']:.3f}"
        + f"-sz={run_config['optimizer_config']['scheduler_params']['step_size']}"
        + "-{epoch:02d}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        gpus=run_config["ngpus"] if torch.cuda.is_available() else 0,
        max_epochs=run_config["max_epochs"],
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(
        model,
        train_dataloaders=split2loader[run_config.get("train_on")],
        val_dataloaders=[
            split2loader[x] for x in run_config.get("validate_on")
        ],
    )
    # Test on test set with model for lowest validation accuracy
    # test_result = trainer.test(
    #     ckpt_path="best",
    #     dataloaders=split2loader[run_config.get("test_on")],
    #     verbose=False,
    # )
    # print(trainer.callback_metrics)


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "validation_loss"},
        "parameters": {
            "sched_step_size": {"values": [5, 10, 15]},
            "weight_decay": {"max": 0.01, "min": 0.0001},
            "lr": {"max": 0.01, "min": 0.0001},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="pcam")

    # Start sweep job.
    wandb.agent(sweep_id, function=evaluate_model, count=2)

    # evaluate_model()
    # TODO: implement step scheduler and vary steps
