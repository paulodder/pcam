import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl


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
