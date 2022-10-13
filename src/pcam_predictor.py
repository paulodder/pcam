import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy
from torch.nn import functional as F
import numpy as np

from src.densenet import fA_P4DenseNet, fA_P4MDenseNet, P4MDenseNet, P4DenseNet

NET_STR2INIT_FUNC = {
    "fA_P4DenseNet": fA_P4DenseNet,
    "fA_P4MDenseNet": fA_P4MDenseNet,
    "P4MDenseNet": P4MDenseNet,
    "P4DenseNet": P4DenseNet,
    # "GResNet18": GResNet18,
}


class PCAMPredictor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model_config"]
        self.optimizer_config = config["optimizer_config"]

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
        self.auroc = {
            "train": AUROC(num_classes=2, pos_label=1),
            "val": AUROC(num_classes=2, pos_label=1),
            "test": AUROC(num_classes=2, pos_label=1),
        }
        self.acc = {"train": Accuracy(), "val": Accuracy(), "test": Accuracy()}

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
        loss, acc, auc = self.forward(batch, mode="train")
        # wandb.log({"loss": loss, "acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, auc = self.forward(batch, mode="val")
        return {
            f"loss": loss,
            f"acc": acc,
            f"auc": auc,
        }

    def validation_epoch_end(self, outputs):
        if type(outputs[0]) == dict:
            outputs = [outputs]

        for dataloader_name, output in zip(
            self.config["validate_on"], outputs
        ):
            acc = np.mean([tmp["acc"] for tmp in output])
            auc = np.mean([tmp["auc"] for tmp in output])
            loss = np.mean([tmp["loss"].cpu() for tmp in output])
            wandb.log(
                {
                    f"{dataloader_name}_acc": acc,
                    f"{dataloader_name}_auc": acc,
                    f"{dataloader_name}_loss": loss,
                }
            )
            if dataloader_name == "validation":
                self.log("val_loss", loss)
                self.val_losses.append(loss)

    def test_step(self, batch, batch_idx):
        loss, acc, auc = self.forward(batch, mode="test")

        return {
            "loss": loss,
            "acc": acc,
            "auc": auc,
        }

    def test_epoch_end(self, outputs):
        acc = np.mean([tmp["acc"] for tmp in outputs])
        auc = np.mean([tmp["auc"] for tmp in outputs])
        loss = np.mean([tmp["loss"].cpu() for tmp in outputs])
        wandb.log({"test_acc": acc, "test_auc": auc, "test_loss": loss})
        self.log("test_acc", acc)

    def forward(self, data, mode="train"):
        x, y = data
        outp = self.model(x)
        y_pred_proba = F.softmax(outp)
        loss = self.loss_module(y_pred_proba, y)
        y_pred_proba = y_pred_proba.cpu().detach()
        y = y.cpu().detach().int()
        auc = self.auroc[mode](y_pred_proba, y)
        acc = self.acc[mode](y_pred_proba, y)
        return loss, acc, auc
