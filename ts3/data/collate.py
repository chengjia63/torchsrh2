import torch


class SlideEmbeddingMILCollator:
    def __call__(self, batch):
        return {
            "path": [sample["path"] for sample in batch],
            "embeddings": [sample["embeddings"] for sample in batch],
            "coords": [sample["coords"] for sample in batch],
            "label": torch.stack([sample["label"] for sample in batch]).long(),
        }
