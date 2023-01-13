from abc import ABC, abstractmethod
from collections import defaultdict

import sklearn.metrics as sk_metrics
import numpy as np
import torch

from .. import utils


class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def value(self, *args, **kwargs):
        pass

    def on_partition_start(self, *args, **kwargs):
        pass

    def on_partition_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass


class MetricWithStorage(Metric, ABC):
    def __init__(self, **kwargs):
        super(MetricWithStorage, self).__init__(**kwargs)
        self._metric_storage = defaultdict(dict)

    def value(self, partition: str, epoch: int, **kwargs):
        if partition not in self._metric_storage:
            raise ValueError(f'No evaluations are found for {partition=}')
        dct = self._metric_storage[partition]
        if epoch not in dct:
            raise ValueError(f'No evaluation is found for {partition=} and {epoch=}')
        return dct[epoch]

    def _store(self, partition: str, epoch: int, value):
        self._metric_storage[partition][epoch] = value


class Accuracy(MetricWithStorage):
    """ Accuracy metric. Works in both binary and multiclass classification settings.
    """
    def __init__(self, output_key: str = 'pred', threshold: float = 0.5, one_hot: bool = False,
                 plus_minus_one_binary: bool = False, **kwargs):
        """
        :param threshold: in the case of binary classification what threshold to use
        :param one_hot: whether the labels is in one-hot encoding
        :param plus_minus_one_binary: in case of binary labels, this specifies whether it is (0 vs 1) or (-1 vs +1)
        """
        super(Accuracy, self).__init__(**kwargs)
        self.output_key = output_key
        self.threshold = threshold
        self.one_hot = one_hot
        self.plus_minus_one_binary = plus_minus_one_binary

        # initialize and use later
        self._iter_accuracies = defaultdict(list)

    @property
    def name(self) -> str:
        return "accuracy"

    def on_partition_start(self, partition, **kwargs):
        self._iter_accuracies[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        accuracy = np.mean(self._iter_accuracies[partition])
        self._store(partition=partition, epoch=epoch, value=accuracy)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def compute_metric(self, preds, labels):
        if preds.shape[-1] > 1:
            # multiple class
            pred = utils.to_numpy(preds).argmax(axis=1).astype(np.int32)
        else:
            # binary classification
            pred = utils.to_numpy(preds.squeeze(dim=-1) > self.threshold).astype(np.int32)
            if self.plus_minus_one_binary:
                pred = 2 * pred - 1

        labels = utils.to_numpy(labels).astype(np.int32)
        if self.one_hot:
            labels = np.argmax(labels, axis=1)
        else:
            labels = labels.reshape(pred.shape)
        return (pred == labels).astype(np.float32).mean()

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        value = self.compute_metric(preds=outputs[self.output_key],
                                    labels=batch_labels[0])
        self._iter_accuracies[partition].append(value)


class MulticlassScalarAccuracy(MetricWithStorage):
    """ Accuracy metric in case when the output is a single scalar, while num_classes > 2.
    """
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(MulticlassScalarAccuracy, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._iter_accuracies = defaultdict(list)

    @property
    def name(self) -> str:
        return "accuracy"

    def on_partition_start(self, partition, **kwargs):
        self._iter_accuracies[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        accuracy = np.mean(self._iter_accuracies[partition])
        self._store(partition=partition, epoch=epoch, value=accuracy)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        out = outputs[self.output_key]
        assert out.shape[-1] == 1
        pred = utils.to_numpy(torch.round(out)).astype(np.int32)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int32).reshape(pred.shape)
        self._iter_accuracies[partition].append((pred == batch_labels).astype(np.float32).mean())


class ROCAUC(MetricWithStorage):
    """ ROC AUC for binary classification setting.
    """
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(ROCAUC, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._score_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "ROC AUC"

    def on_partition_start(self, partition, **kwargs):
        self._score_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        scores = torch.cat(self._score_storage[partition], dim=0)
        try:
            auc = sk_metrics.roc_auc_score(y_true=utils.to_numpy(labels),
                                           y_score=utils.to_numpy(scores))
        except ValueError:
            auc = np.nan
        self._store(partition=partition, epoch=epoch, value=auc)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", auc, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        assert pred.shape[-1] == 1
        self._score_storage[partition].append(pred.squeeze(dim=-1))
        self._label_storage[partition].append(batch_labels[0])


class TopKAccuracy(MetricWithStorage):
    def __init__(self, k, output_key: str = 'pred', **kwargs):
        super(TopKAccuracy, self).__init__(**kwargs)
        self.k = k
        self.output_key = output_key

        # initialize and use later
        self._iter_accuracies = defaultdict(list)

    @property
    def name(self) -> str:
        return f"top{self.k}_accuracy"

    def on_partition_start(self, partition, **kwargs):
        self._iter_accuracies[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        accuracy = np.mean(self._iter_accuracies[partition])
        self._store(partition=partition, epoch=epoch, value=accuracy)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(outputs[self.output_key])
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int32)

        topk_predictions = np.argsort(-pred, axis=1)[:, :self.k]
        batch_labels = batch_labels.reshape((-1, 1)).repeat(self.k, axis=1)
        topk_correctness = (np.sum(topk_predictions == batch_labels, axis=1) >= 1)

        self._iter_accuracies[partition].append(topk_correctness.astype(np.float32).mean())


class F1Score(MetricWithStorage):
    """ F1 score for binary classification setting.
    """
    def __init__(self, threshold: float = 0.5, output_key: str = 'pred', **kwargs):
        super(F1Score, self).__init__(**kwargs)
        self.threshold = threshold
        self.output_key = output_key

        # initialize and use later
        self._score_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "F1"

    def on_partition_start(self, partition, **kwargs):
        self._score_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        scores = torch.cat(self._score_storage[partition], dim=0)

        f1 = sk_metrics.f1_score(y_true=utils.to_numpy(labels),
                                 y_pred=utils.to_numpy(scores),
                                 zero_division=0)
        self._store(partition=partition, epoch=epoch, value=f1)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", f1, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        assert pred.shape[-1] == 1
        self._score_storage[partition].append((pred.squeeze(dim=-1) > self.threshold).long())
        self._label_storage[partition].append(batch_labels[0])


class PRCAUC(MetricWithStorage):
    """ PRC AUC for binary classification setting.
    """
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(PRCAUC, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._score_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "PRC AUC"

    def on_partition_start(self, partition, **kwargs):
        self._score_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        scores = torch.cat(self._score_storage[partition], dim=0)

        (precisions, recalls, thresholds) = sk_metrics.precision_recall_curve(y_true=utils.to_numpy(labels),
                                                                              probas_pred=utils.to_numpy(scores))
        try:
            auc = sk_metrics.auc(recalls, precisions)
        except ValueError:
            auc = np.nan
        self._store(partition=partition, epoch=epoch, value=auc)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", auc, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        assert pred.shape[-1] == 1
        self._score_storage[partition].append(pred.squeeze(dim=-1))
        self._label_storage[partition].append(batch_labels[0])


class MacroF1Score(MetricWithStorage):
    """ Macro averaged F1 score for multiclass classification setting.
    """
    def __init__(self, num_classes: int, output_key: str = 'pred', **kwargs):
        super(MacroF1Score, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.output_key = output_key

        # initialize and use later
        self._preds_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "Macro F1"

    def on_partition_start(self, partition, **kwargs):
        self._preds_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        preds = torch.cat(self._preds_storage[partition], dim=0)

        f1s = []
        for c in range(self.num_classes):
            cur_labels = (labels == c).long()
            cur_pred = (preds.argmax(dim=1) == c).long()
            cur_f1 = sk_metrics.f1_score(y_true=utils.to_numpy(cur_labels),
                                         y_pred=utils.to_numpy(cur_pred),
                                         zero_division=0)
            f1s.append(cur_f1)

        avg_f1 = np.mean(f1s)
        self._store(partition=partition, epoch=epoch, value=avg_f1)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", avg_f1, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        self._preds_storage[partition].append(pred)
        self._label_storage[partition].append(batch_labels[0])


class MacroROCAUC(MetricWithStorage):
    """ Macro averaged ROC AUC score for multiclass classification setting.
    """
    def __init__(self, num_classes: int, output_key: str = 'pred', **kwargs):
        super(MacroROCAUC, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.output_key = output_key

        # initialize and use later
        self._probs_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "Macro ROC AUC"

    def on_partition_start(self, partition, **kwargs):
        self._probs_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        preds = torch.cat(self._probs_storage[partition], dim=0)

        aucs = []
        for c in range(self.num_classes):
            cur_labels = (labels == c).long()
            cur_scores = preds[:, c]
            try:
                cur_auc = sk_metrics.roc_auc_score(y_true=utils.to_numpy(cur_labels),
                                                   y_score=utils.to_numpy(cur_scores))
            except ValueError:
                cur_auc = np.nan

            aucs.append(cur_auc)

        avg_auc = np.nanmean(aucs)
        self._store(partition=partition, epoch=epoch, value=avg_auc)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", avg_auc, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        self._probs_storage[partition].append(torch.softmax(pred, dim=1))
        self._label_storage[partition].append(batch_labels[0])


class MacroPRCAUC(MetricWithStorage):
    """ Macro averaged PRC AUC score for multiclass classification setting.
    """
    def __init__(self, num_classes: int, output_key: str = 'pred', **kwargs):
        super(MacroPRCAUC, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.output_key = output_key

        # initialize and use later
        self._probs_storage = defaultdict(list)
        self._label_storage = defaultdict(list)

    @property
    def name(self) -> str:
        return "Macro PRC AUC"

    def on_partition_start(self, partition, **kwargs):
        self._probs_storage[partition] = []
        self._label_storage[partition] = []

    def on_partition_end(self, partition, epoch, tensorboard, **kwargs):
        labels = torch.cat(self._label_storage[partition], dim=0)
        preds = torch.cat(self._probs_storage[partition], dim=0)

        aucs = []
        for c in range(self.num_classes):
            cur_labels = (labels == c).long()
            cur_scores = preds[:, c]
            (precisions, recalls, thresholds) = sk_metrics.precision_recall_curve(
                y_true=utils.to_numpy(cur_labels),
                probas_pred=utils.to_numpy(cur_scores)
            )
            cur_auc = sk_metrics.auc(recalls, precisions)
            aucs.append(cur_auc)

        avg_auc = np.nanmean(aucs)
        self._store(partition=partition, epoch=epoch, value=avg_auc)
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", avg_auc, epoch)

    def on_iteration_end(self, outputs, batch_labels, partition, **kwargs):
        pred = outputs[self.output_key]
        self._probs_storage[partition].append(torch.softmax(pred, dim=1))
        self._label_storage[partition].append(batch_labels[0])
