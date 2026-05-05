import math
import random
import logging
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LabelIndexTensorTransform:
    def __init__(self, class_order: list):
        self.class_to_idx_ = {c: i for i, c in enumerate(class_order)}

    def __call__(self, label):
        if label not in self.class_to_idx_:
            raise KeyError(f"Unknown label {label!r}")
        return torch.tensor(self.class_to_idx_[label], dtype=torch.long)


class BalanceableBaseDataset(Dataset):
    def __init__(
        self,
        label_key: str = "label",
        eps: float = 1.0e-6,
        class_order: Optional[list] = None,
    ):
        super().__init__()
        self.label_key_ = label_key
        self.classes_: Optional[list] = class_order
        self.class_to_idx_: Optional[dict] = None
        self.weights_ = None
        self.eps_ = eps

    def get_instance_label(self, inst):
        if self.label_key_ not in inst:
            raise KeyError(
                f"Instance is missing required label key {self.label_key_!r}"
            )
        return inst[self.label_key_]

    def process_classes(self):
        all_labels = set(self.get_instance_label(i) for i in self.instances_)
        if self.classes_ is not None:
            specified = set(self.classes_)
            if specified != all_labels:
                raise ValueError(
                    f"class_order {sorted(specified)} does not match labels in data {sorted(all_labels)}"
                )
        else:
            classes = list(all_labels)
            if len(set(map(type, classes))) == 1:
                self.classes_ = sorted(classes)
            else:
                sort_class_idx = np.argsort([str(c) for c in classes]).tolist()
                self.classes_ = [classes[i] for i in sort_class_idx]

        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: %s", self.classes_)

    def get_count(self):
        if not self.classes_ or self.class_to_idx_ is None:
            self.process_classes()

        all_labels = [
            self.class_to_idx_[self.get_instance_label(i)] for i in self.instances_
        ]

        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(self.classes_))])
        logging.info("Count: %s", count)
        return count

    def get_weights(self):
        count = self.get_count()
        inv_count = 1.0 / (count + self.eps_)
        inv_count[count == 0] = 0
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: %s", self.weights_)
        return self.weights_

    def replicate_balance_instances(self):
        logging.info("replicating instances to balance each class")
        self.process_classes()
        count = self.get_count()
        val_sample = int(max(count))
        random.shuffle(self.instances_)
        all_repl_instances = []

        for label in self.classes_:
            inst_l = [i for i in self.instances_ if self.get_instance_label(i) == label]
            n_rep = math.ceil(val_sample / len(inst_l))
            all_repl_instances.extend((inst_l * n_rep)[:val_sample])

        self.instances_ = all_repl_instances
