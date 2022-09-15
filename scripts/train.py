from argparse import ArgumentParser

import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from src import utils
from src.resnet import ResNet, ResBlock

# X, Y = next(iter(utils.get_dataloader("test")))
# X = X.to("cuda:0")
# Y = Y.to("cuda:0")


def get_parser():
    parser = ArgumentParser()
    # dataloader arguments
    parser.add_argument("--batch_size", default=128, type=int)
    # optimizer arguments
    parser.add_argument("--lr", default=0.001)
    parser.add_argument(
        "--weight_decay",
        default=0.01,
    )
    # model args
    parser.add_argument(
        "--block",
        default=ResBlock,
    )
    parser.add_argument(
        "--layers",
        default=[3, 4, 6, 3],
        type=int,
    )
    # run arguments
    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
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
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.model = ResNet(**model_config)
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
        wandb.log({"test_acc": acc, "test_loss": loss})

    def forward(self, data, mode="train"):
        x, y = data
        y_pred_proba = self.model(x)
        loss = self.loss_module(y_pred_proba, y)
        acc = ((y_pred_proba > 0.5).int() == y.int()).float().mean()
        return loss, acc


if __name__ == "__main__":
    # args = parse_args()
    args = get_mock_args()
    wandb_config = {
        "dataset_config": {"batch_size": args.batch_size},
        "optimizer_config": {"weight_decay": args.weight_decay, "lr": args.lr},
        "model_config": {"in_channels": 3, "resblock_cls": ResBlock},
    }
    wandb.init(
        project="pcam",
        config=wandb_config,
    )
    # get dataloaders
    split2loader = {
        split: utils.get_dataloader(split, **wandb_config["dataset_config"])
        for split in ["test", "validation"]
    }
    model = PCAMPredictor(
        wandb_config["model_config"], wandb_config["optimizer_config"]
    )
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, split2loader["test"], split2loader["validation"])
    test_result = trainer.test(model, split2loader["test"], verbose=False)