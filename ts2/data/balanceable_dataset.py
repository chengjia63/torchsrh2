import math
import random
import logging
from collections import Counter

import numpy as np

import torch
from torch.utils.data import Dataset


class PrimaryLabelFunc():

    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        return x[self.idx]


class BalanceableBaseDataset(Dataset):
    """Balanceable Base Dataset.

    Datasets that allows data from each class to be balanced

    Attributes:
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """

    def __init__(self,
                 primary_label_func: callable = PrimaryLabelFunc(0),
                 eps=1.0e-6,
                 classes=None,
                 class_to_idx=None):
        super().__init__()

        self.primary_label_func_ = primary_label_func

        self.classes_ = classes
        self.class_to_idx_ = class_to_idx
        self.weights_ = None
        self.eps_ = eps

    def process_classes(self):
        """Look for all the labels in the dataset

        Creates the classes_, and class_to_idx_ attributes"""

        if not self.classes_:
            all_labels = [i["label"] for i in self.instances_]
            classes = set(all_labels)

            if len(set(map(type, classes))) == 1:
                self.classes_ = sorted(classes)
            else:
                classes = list(classes)
                sort_class_idx = np.argsort([str(c) for c in classes]).tolist()
                self.classes_ = [classes[i] for i in sort_class_idx]

            self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}

        logging.info("Labels: {}".format(self.classes_))

    def get_count(self):
        # Count number of slides in each class
        if not self.classes_: self.process_classes()
        all_labels = [self.class_to_idx_[i["label"]] for i in self.instances_]

        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(self.classes_))])
        logging.info("Count: {}".format(count))
        return count

    def get_weights(self):
        """Count number of instances for each class, and computes weights"""
        # Get classes and count number of slides in each class
        count = self.get_count()

        # Compute weights
        inv_count = 1.0 / (count + self.eps_)
        inv_count[count == 0] = 0
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))

        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class"""
        logging.info("replicating instances to balance each class")

        # Get classes
        self.process_classes()
        count = self.get_count()
        val_sample = int(max(count))
        random.shuffle(self.instances_)
        all_repl_instances = []

        for l in self.classes_:
            inst_l = [i for i in self.instances_ if i["label"] == l]
            n_rep = math.ceil(val_sample / len(inst_l))
            all_repl_instances.extend((inst_l * n_rep)[:val_sample])

        self.instances_ = all_repl_instances
