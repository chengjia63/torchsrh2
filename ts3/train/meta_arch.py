import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics


class CORALLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def coral_levels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1)
        thresholds = torch.arange(self.num_classes - 1, device=labels.device)
        return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        expected_dim = self.num_classes - 1
        levels = self.coral_levels(labels).to(dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, levels)


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels.long().view(-1))


class GatedABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        attention_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)

    def forward(self, embeddings: torch.Tensor):
        encoded = self.instance_encoder(embeddings)  # [N, H]

        attention_logits = self.attention_w(
            self.attention_v(encoded) * self.attention_u(encoded)
        ).transpose(
            1, 0
        )  # [1, N]

        attention = F.softmax(attention_logits, dim=1)  # [1, N]
        pooled = torch.mm(attention, encoded).squeeze(0)  # [H]

        return {
            "pooled_embeddings": pooled,
            "attention": attention.squeeze(0),
            "encoded_embeddings": encoded,
        }


class SharedCORALHead(nn.Module):
    """
    Strict CORAL-style head:
    one shared scalar score + threshold-specific biases.

    Produces logits for:
        P(y > 0), P(y > 1), ..., P(y > K-2)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2 for CORAL, got {num_classes}"
            )

        self.num_classes = num_classes
        self.score = nn.Linear(input_dim, 1, bias=False)
        self.coral_bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor):
        score = self.score(x).squeeze(-1)  # scalar or [B]
        logits = score.unsqueeze(-1) + self.coral_bias  # [K - 1] or [B, K - 1]

        return {
            "logits": logits,
            "score": score,
        }


class MultiHeadCORALHead(nn.Module):
    """
    Flexible cumulative ordinal head:
    separate binary classifier per ordinal threshold.

    Produces logits for:
        P(y > 0), P(y > 1), ..., P(y > K-2)
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"num_classes must be at least 2 for ordinal classification, got {num_classes}"
            )

        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes - 1)

    def forward(self, x: torch.Tensor):
        logits = self.classifier(x)  # [K - 1] or [B, K - 1]

        # Optional scalar summary for monitoring only.
        # This is not a true latent CORAL score.
        score = logits.mean(dim=-1)

        return {
            "logits": logits,
            "score": score,
        }


class CrossEntropyHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor):
        logits = self.classifier(x)
        return {
            "logits": logits,
        }


class MILOrdinalModel(nn.Module):
    def __init__(self, mil: nn.Module, head: nn.Module):
        super().__init__()
        self.mil = mil
        self.head = head

    def forward(self, embeddings: torch.Tensor):
        out = self.mil(embeddings)
        head_out = self.head(out["pooled_embeddings"])

        return {
            **out,
            **head_out,
        }


class SlideABMILModule(pl.LightningModule):
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
        self.num_classes = num_classes
        self.criterion = criterion(num_classes=num_classes)

        self.train_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        )
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes
        )

        self.train_pearson = torchmetrics.regression.PearsonCorrCoef()
        self.val_pearson = torchmetrics.regression.PearsonCorrCoef()
        self.train_spearman = torchmetrics.regression.SpearmanCorrCoef()
        self.val_spearman = torchmetrics.regression.SpearmanCorrCoef()
        self.validation_outputs = []

    def forward(self, batch):
        bag_outputs = [
            self.model(embeddings.float()) for embeddings in batch["embeddings"]
        ]

        logits = torch.stack([output["logits"] for output in bag_outputs], dim=0)
        outputs = {
            "logits": logits,
            "attention": [output["attention"] for output in bag_outputs],
            "pooled_embeddings": torch.stack(
                [output["pooled_embeddings"] for output in bag_outputs], dim=0
            ),
        }
        if "score" in bag_outputs[0]:
            outputs["raw_score"] = torch.stack(
                [output["score"] for output in bag_outputs], dim=0
            )
        return outputs

    def _is_coral_logits(self, logits):
        return isinstance(self.model.head, (SharedCORALHead, MultiHeadCORALHead))

    def _is_ce_logits(self, logits):
        return isinstance(self.model.head, CrossEntropyHead)

    def _score(self, logits):
        if self._is_coral_logits(logits):
            return torch.sigmoid(logits).sum(dim=1)
        if self._is_ce_logits(logits):
            return logits.argmax(dim=1).float()
        self._raise_unrecognized_logits(logits)

    def _correlation_score(self, outputs):
        if self._is_coral_logits(outputs["logits"]) and "raw_score" in outputs:
            return outputs["raw_score"]
        return self._score(outputs["logits"])

    def _pred_label(self, logits):
        if self._is_coral_logits(logits):
            return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()
        if self._is_ce_logits(logits):
            return logits.argmax(dim=1)
        self._raise_unrecognized_logits(logits)

    def _score_pred_label(self, score):
        return score.round().clamp(0, self.num_classes - 1).long()

    def _raise_unrecognized_logits(self, logits):
        raise ValueError(
            f"Expected model.head to be a CORAL or CE head, got "
            f"{type(self.model.head).__name__}"
        )

    def _label(self, batch):
        return batch["label"].to(device=self.device, dtype=torch.long).view(-1)

    def _score_target(self, batch):
        return batch["label"].to(device=self.device, dtype=torch.float32).view(-1)

    def training_step(self, batch, _batch_idx):
        outputs = self(batch)

        label = self._label(batch)
        loss = self.criterion(outputs["logits"], label)
        score = self._score(outputs["logits"])
        correlation_score = self._correlation_score(outputs)
        pred_label = self._pred_label(outputs["logits"])

        score_target = self._score_target(batch)

        self.train_acc.update(pred_label, label)
        self.train_pearson.update(correlation_score, score_target)
        self.train_spearman.update(correlation_score, score_target)

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
            "train/pearson",
            self.train_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/spearman",
            self.train_spearman,
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
        label = self._label(batch)
        self.validation_outputs.append(
            {
                "logits": outputs["logits"].detach(),
                "raw_score": (
                    outputs["raw_score"].detach() if "raw_score" in outputs else None
                ),
                "label": label.detach(),
            }
        )

    def on_validation_epoch_end(self):
        logits = torch.cat([output["logits"] for output in self.validation_outputs])
        raw_scores = [output["raw_score"] for output in self.validation_outputs]
        label = torch.cat([output["label"] for output in self.validation_outputs])
        self.validation_outputs.clear()

        logits = self._all_gather_variable_length(logits)
        raw_score = (
            self._all_gather_variable_length(torch.cat(raw_scores))
            if raw_scores and raw_scores[0] is not None
            else None
        )
        label = self._all_gather_variable_length(label)

        loss = self.criterion(logits, label)
        score = self._score(logits)
        correlation_score = raw_score if raw_score is not None else score
        pred_label = self._pred_label(logits)
        score_target = label.float()

        self.val_acc.update(pred_label, label)
        self.val_pearson.update(correlation_score, score_target)
        self.val_spearman.update(correlation_score, score_target)

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
            "val/pearson",
            self.val_pearson,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/spearman",
            self.val_spearman,
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
        score = self._score(outputs["logits"])
        prediction = {
            "path": batch["path"],
            "label": batch["label"],
            "logits": outputs["logits"],
            "score": score,
            "pred_label": self._pred_label(outputs["logits"]),
            "attention": outputs["attention"],
            "pooled_embeddings": outputs["pooled_embeddings"],
        }
        if "raw_score" in outputs:
            prediction["raw_score"] = outputs["raw_score"]
        if self._is_coral_logits(outputs["logits"]):
            prediction["score_pred_label"] = self._score_pred_label(score)
        return prediction

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
