import torch
import pytorch_lightning as pl
from dinov2.models import build_model
import dinov2.utils.utils as dinov2_utils


class Dinov2EvalSystem(pl.LightningModule):

    def __init__(self, model_hyperparams, pretrained_weights):
        super().__init__()
        self.teacher_backbone, _ = build_model(model_hyperparams,
                                               only_teacher=True)
        dinov2_utils.load_pretrained_weights(self.teacher_backbone,
                                             pretrained_weights, "teacher")
        self.teacher_backbone.eval()

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch["image"].shape[1] == 1

        emb = self.teacher_backbone(batch["image"][:, 0, ...])
        results = {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": emb
        }

        return results
