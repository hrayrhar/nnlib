"""
Wrappers around datasets.
"""
import logging

import numpy as np
import torch.utils.data
import torch.nn.functional as F


class TwoAugmentationWrapper(torch.utils.data.Dataset):
    """ Takes a dataset instance and creates another dataset which returns two data augmentations per example.
    """
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(TwoAugmentationWrapper, self).__init__()
        self.dataset = dataset
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # generate two views from one example and return both as a single sample
        (x1, y) = self.dataset[idx]
        (x2, y) = self.dataset[idx]
        return [x1, x2], y


class SemiSupervisedWrapper(torch.utils.data.Dataset):
    """ Takes a dataset and creates a semi-supervised dataset.
    :NOTE: the "label" for unlabeled examples will be -1.
    """
    def __init__(self, dataset, labeled_count_per_class):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param labeled_count_per_class: number of labeled examples per class
        """
        self.dataset = dataset
        self.labeled_count_per_class = labeled_count_per_class

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

        # mark unlabeled examples
        self.is_labeled = np.zeros(len(dataset), dtype=np.bool)
        all_labels = [y for (x, y) in self.dataset]
        all_labels = np.array(all_labels)
        num_classes = np.max(all_labels)

        for y in range(num_classes):
            cur_label_indices = np.where(all_labels == y)[0]
            choose_cnt = min(labeled_count_per_class, len(cur_label_indices))
            labeled_indices = np.random.choice(cur_label_indices, size=choose_cnt, replace=False)
            for idx in labeled_indices:
                self.is_labeled[idx] = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.is_labeled[idx]:
            return x, y
        return x, -1


class LabelSubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, which_labels=(0, 1)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which labels to use
        """
        super(LabelSubsetWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics
        self.valid_indices = [idx for idx, (x, y) in enumerate(dataset) if y in which_labels]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.valid_indices[idx]]
        assert y in self.which_labels
        new_y = self.which_labels.index(y)
        return x, torch.tensor(new_y, dtype=torch.long)


BinaryDatasetWrapper = LabelSubsetWrapper  # shortcut


class LabelSelectorWrapper(torch.utils.data.Dataset):
    """ Select a subset of label in case it is an array. """

    def __init__(self, dataset, which_labels):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which coordinates of label array to select
        """
        super(LabelSelectorWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y[self.which_labels]


class OneHotLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param num_classes: number of classes
        """
        super(OneHotLabelWrapper, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = torch.tensor(y, dtype=torch.long)
        y = F.one_hot(y, num_classes=self.num_classes)
        return x, y


class ReturnSampleIndexWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(ReturnSampleIndexWrapper, self).__init__()
        self.dataset = dataset
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return (x, idx), y


class SubsetDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices=None, include_indices=None):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(SubsetDataWrapper, self).__init__()

        if exclude_indices is None:
            assert include_indices is not None
        if include_indices is None:
            assert exclude_indices is not None

        self.dataset = dataset

        if include_indices is not None:
            self.include_indices = include_indices
        else:
            S = set(exclude_indices)
            self.include_indices = [idx for idx in range(len(dataset)) if idx not in S]

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]


class ResizeImagesWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, size=(224, 224)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(ResizeImagesWrapper, self).__init__()
        self.dataset = dataset
        self.size = size

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = F.interpolate(x.unsqueeze(dim=0), size=self.size, mode='bilinear')[0]
        return x, y


class CacheDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(CacheDatasetWrapper, self).__init__()
        self.dataset = dataset

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

        # create cache
        logging.warning(f"Caching dataset {dataset.dataset_name}. Assuming data augmentation is disabled.")
        self._cached_dataset = [p for p in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._cached_dataset[idx]


class MergeDatasetsWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        """
        :params dataset1 and dataset2: StandardVisionDataset or derivative class instance
        """
        super(MergeDatasetsWrapper, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # record important attributes
        self.dataset_name = f'merge {dataset1.dataset_name} and {dataset2.dataset_name}'
        self.statistics = dataset1.statistics

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx - len(self.dataset1)]


class GrayscaleToColoredWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :params dataset: StandardVisionDataset or derivative class instance
        """
        super(GrayscaleToColoredWrapper, self).__init__()
        self.dataset = dataset

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = (dataset.statistics[0].repeat(3), dataset.statistics[1].repeat(3))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x.repeat(3, 1, 1), y


class UniformNoiseWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, error_prob, num_classes, seed=42):
        """
        :params dataset: StandardVisionDataset or derivative class instance
        :params error_prob: the probability that a label is switched
        Assumes that the labels are integers
        """
        super(UniformNoiseWrapper, self).__init__()
        self.dataset = dataset
        self.error_prob = error_prob
        self.num_classes = num_classes
        self.seed = seed

        # prepare samples
        self.ys = []
        self.is_corrupted = np.zeros(len(dataset), dtype=np.bool)
        np.random.seed(seed)
        for idx in range(len(dataset)):
            y = torch.tensor(dataset[idx][1]).item()
            if np.random.rand() < error_prob:
                self.is_corrupted[idx] = True
                while True:
                    new_y = np.random.choice(num_classes)
                    if new_y != y:
                        y = new_y
                        break
            self.ys.append(y)

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.ys[idx]
