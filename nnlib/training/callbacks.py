from abc import ABC, abstractmethod
import os

import numpy as np

from .. import utils
from .metrics import Metric


class Callback(ABC):
    def __init__(self, **kwargs):
        pass

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass

    @abstractmethod
    def call(self, *args, **kwargs):
        pass


class SaveBestWithMetric(Callback):
    def __init__(self,
                 metric: Metric,
                 partition: str = 'val',
                 direction: str = 'max',
                 **kwargs):

        assert direction in ['min', 'max']
        super(SaveBestWithMetric, self).__init__(**kwargs)
        self.metric = metric
        self.partition = partition
        self.direction = direction

        if self.direction == 'max':
            self._best_result_so_far = -np.inf
        else:
            self._best_result_so_far = +np.inf

    def call(self, epoch, model, optimizer, scheduler, log_dir, **kwargs):
        result = self.metric.value(partition=self.partition, epoch=epoch)

        update = (self.direction == 'max' and result > self._best_result_so_far) or \
                 (self.direction == 'min' and result < self._best_result_so_far)

        if update:
            self._best_result_so_far = result
            print(f"This is the best {self.partition} result w.r.t. {self.metric.name} so far. Saving the model ...")
            utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
                       path=os.path.join(log_dir, 'checkpoints', f"best_{self.partition}_{self.metric.name}.mdl"))

            # save the validation result for doing model selection later
            with open(os.path.join(log_dir, f"best_{self.partition}_{self.metric.name}.txt"), 'w') as f:
                f.write(f"{result}\n")


class EarlyStoppingWithMetric(Callback):
    def __init__(self,
                 metric: Metric,
                 stopping_param: int = 50,
                 partition: str = 'val',
                 direction: str = 'max',
                 **kwargs):

        assert direction in ['min', 'max']
        super(EarlyStoppingWithMetric, self).__init__(**kwargs)
        self.metric = metric
        self.stopping_param = stopping_param
        self.partition = partition
        self.direction = direction

        if self.direction == 'max':
            self._best_result_so_far = -np.inf
        else:
            self._best_result_so_far = +np.inf

        self._best_result_epoch = -1

    def call(self, epoch, **kwargs) -> bool:
        """ Checks whether the training should be finished. Returning True corresponds to finishing. """
        result = self.metric.value(partition=self.partition, epoch=epoch)

        update = (self.direction == 'max' and result > self._best_result_so_far) or \
                 (self.direction == 'min' and result < self._best_result_so_far)

        if update:
            self._best_result_so_far = result
            self._best_result_epoch = epoch

        return epoch - self._best_result_epoch > self.stopping_param
