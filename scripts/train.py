from argparse import ArgumentParser

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from src import utils


def get_parser():
    parser = ArgumentParser()
    # dataloader arguments
    parser.add_argument("--batch_size", default=64, type=int)
    # optimizer arguments
    parser.add_argument("--lr", default=0.001)
    parser.add_argument(
        "--weight_decay",
        default=0.01,
    )
    return parser


def get_mock_args(overwrite_kwargs=dict()):
    "for dev purposes"
    parser = get_parser()
    args, _ = parser.parse_known_args(None, None)

    class Args(object):
        pass

    mock_args = Args()
    for name, default_value in args._get_kwargs():
        setattr(mock_args, name, overwrite_kwargs.get(name, default_value))
    return mock_args


def parse_args():
    return get_parser().parse_args()


class PCAMPredictor(pl.LightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config,
    ):
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.model = None  # (**model_config)
        self.loss_module = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config["lr"],
            weight_decay=self.optimizer_config["weight_decay"],
        )  # High lr because of small dataset and small model
        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        wandb.log({"loss": loss, "acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        return {
            "loss": loss,
            "acc": acc,
        }

    def validation_epoch_end(self, outputs):
        acc = np.mean([tmp["acc"].cpu() for tmp in outputs])
        loss = np.mean([tmp["loss"].cpu() for tmp in outputs])
        wandb.log({"validation_acc": acc, "validation_loss": loss})

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="test")
        return {
            "loss": loss,
            "acc": acc,
        }

    def test_epoch_end(self, outputs):
        acc = np.mean([tmp["acc"].cpu() for tmp in outputs])
        loss = np.mean([tmp["loss"].cpu() for tmp in outputs])
        wandb.log({"validation_acc": acc, "validation_loss": loss})

    def forward(self, data, mode="train"):
        y_pred_proba = self.model(data.x)
        loss = self.loss_module(y_pred_proba, data.y)
        acc = (y_pred_proba.round() == data.y).mean()
        return loss, acc


if __name__ == "__main__":
    # args = parse_args()
    args = get_mock_args()
    wandb_config = {
        "dataset_config": {"batch_size": args.batch_size},
        "optimizer_config": {"weight_decay": args.weight_decay, "lr": args.lr},
    }
    wandb.init(
        project="pcam",
        config=wandb_config,
    )
    split2loader = {
        split: utils.get_dataloader(split, **wandb_config["dataset_config"])
        for split in ["train", "test", "validation"]
    }
    model = PCAMPredictor(
        wandb_config["model_config"], wandb_config["optimizer_config"]
    )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, split2loader["train"], split2loader["validation"])
    # train_loader = utils.get_dataloader(
    #     'train', **wandb_config["dataset_config"]
    # )
    # test_loader = utils.get_dataloader(
    #     'test', **wandb_config["dataset_config"]
    # )
