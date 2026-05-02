import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics.classification as tm_cls
import torchmetrics.regression as tm_reg


class SlideABMILOrdinalModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion,
        num_classes: int,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.optimizer_ = optimizer
        self.scheduler_ = scheduler

        self.model = model
        self.model_uses_pe = getattr(self.model, "pe", None) is not None

        self.num_classes = num_classes
        self.criterion = criterion(num_classes=num_classes)

        self.train_acc = tm_cls.MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = tm_cls.MulticlassAccuracy(num_classes=num_classes)

        self.train_raw_score_pearson = tm_reg.PearsonCorrCoef()
        self.val_raw_score_pearson = tm_reg.PearsonCorrCoef()

        self.train_raw_score_spearman = tm_reg.SpearmanCorrCoef()
        self.val_raw_score_spearman = tm_reg.SpearmanCorrCoef()

        self.train_sigmoid_raw_score_pearson = tm_reg.PearsonCorrCoef()
        self.val_sigmoid_raw_score_pearson = tm_reg.PearsonCorrCoef()

        self.train_sigmoid_raw_score_spearman = tm_reg.SpearmanCorrCoef()
        self.val_sigmoid_raw_score_spearman = tm_reg.SpearmanCorrCoef()

        self.validation_outputs = []

    def forward(self, batch):
        if self.model_uses_pe:
            bag_outputs = [
                self.model(embeddings.float(), coords)
                for embeddings, coords in zip(batch["embeddings"], batch["coords"])
            ]
        else:
            bag_outputs = [
                self.model(embeddings.float()) for embeddings in batch["embeddings"]
            ]

        logits = torch.stack([output["logits"] for output in bag_outputs], dim=0)
        outputs = {
            "logits": logits,
            "attention": [output["attention"] for output in bag_outputs],
            "cell_score": [
                output.get(
                    "regressed_embeddings",
                    torch.zeros_like(output["attention"]),
                ).squeeze(-1)
                for output in bag_outputs
            ],
        }
        outputs["raw_score"] = torch.stack(
            [output["score"] for output in bag_outputs], dim=0
        )
        return outputs

    def training_step(self, batch, _batch_idx):
        outputs = self(batch)

        label = batch["label"].to(device=self.device, dtype=torch.long).view(-1)
        loss = self.criterion(outputs["logits"], label)
        probs = torch.sigmoid(outputs["logits"])
        pred_label = (probs > 0.5).sum(dim=1).long()
        sigmoid_raw_score = torch.sigmoid(outputs["raw_score"])

        score_target = label.float()

        self.train_acc.update(pred_label, label)
        self.train_raw_score_pearson.update(outputs["raw_score"], score_target)
        self.train_raw_score_spearman.update(outputs["raw_score"], score_target)
        self.train_sigmoid_raw_score_pearson.update(sigmoid_raw_score, score_target)
        self.train_sigmoid_raw_score_spearman.update(sigmoid_raw_score, score_target)

        batch_size = batch["label"].shape[0]

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/raw_score_pearson",
            self.train_raw_score_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/raw_score_spearman",
            self.train_raw_score_spearman,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/sigmoid_raw_score_pearson",
            self.train_sigmoid_raw_score_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/sigmoid_raw_score_spearman",
            self.train_sigmoid_raw_score_spearman,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs.clear()

    def validation_step(self, batch, _batch_idx):
        outputs = self(batch)
        label = batch["label"].to(device=self.device, dtype=torch.long).view(-1)
        self.validation_outputs.append(
            {
                "logits": outputs["logits"].detach(),
                "raw_score": outputs["raw_score"].detach(),
                "label": label.detach(),
            }
        )

    def on_validation_epoch_end(self):
        logits = torch.cat([output["logits"] for output in self.validation_outputs])
        raw_scores = [output["raw_score"] for output in self.validation_outputs]
        label = torch.cat([output["label"] for output in self.validation_outputs])
        self.validation_outputs.clear()

        logits = self._all_gather_variable_length(logits)
        raw_score = self._all_gather_variable_length(torch.cat(raw_scores))
        label = self._all_gather_variable_length(label)

        loss = self.criterion(logits, label)
        probs = torch.sigmoid(logits)
        pred_label = (probs > 0.5).sum(dim=1).long()
        sigmoid_raw_score = torch.sigmoid(raw_score)
        score_target = label.float()

        self.val_acc.update(pred_label, label)
        self.val_raw_score_pearson.update(raw_score, score_target)
        self.val_raw_score_spearman.update(raw_score, score_target)
        self.val_sigmoid_raw_score_pearson.update(sigmoid_raw_score, score_target)
        self.val_sigmoid_raw_score_spearman.update(sigmoid_raw_score, score_target)

        batch_size = label.shape[0]
        self.log(
            "val/num_datapoints",
            float(batch_size),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            loss,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/acc",
            self.val_acc,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/raw_score_pearson",
            self.val_raw_score_pearson,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/raw_score_spearman",
            self.val_raw_score_spearman,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/sigmoid_raw_score_pearson",
            self.val_sigmoid_raw_score_pearson,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/sigmoid_raw_score_spearman",
            self.val_sigmoid_raw_score_spearman,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def _all_gather_variable_length(self, tensor: torch.Tensor) -> torch.Tensor:
        local_size = torch.tensor(tensor.shape[0], device=tensor.device)
        gathered_sizes = self.all_gather(local_size).view(-1)
        max_size = int(gathered_sizes.max())

        if tensor.shape[0] < max_size:
            pad_shape = (max_size - tensor.shape[0], *tensor.shape[1:])
            padding = torch.zeros(
                pad_shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, padding], dim=0)

        gathered = self.all_gather(tensor)
        if gathered.ndim == tensor.ndim:
            return gathered[: int(local_size)]

        return torch.cat(
            [
                rank_tensor[: int(rank_size)]
                for rank_tensor, rank_size in zip(gathered, gathered_sizes)
            ],
            dim=0,
        )

    def predict_step(self, batch, _batch_idx, _dataloader_idx=0):
        outputs = self(batch)
        probs = torch.sigmoid(outputs["logits"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "logits": outputs["logits"],
            "raw_score": outputs["raw_score"],
            "pred_label": (probs > 0.5).sum(dim=1).long(),
            "attention": outputs["attention"],
            "cell_score": outputs["cell_score"],
        }

    def configure_optimizers(self):
        optimizer = self.optimizer_(
            params=filter(lambda p: p.requires_grad, self.model.parameters())
        )

        if self.scheduler_ is None:
            return [optimizer]

        scheduler = self.scheduler_(optimizer=optimizer)
        return [optimizer], {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr",
        }
