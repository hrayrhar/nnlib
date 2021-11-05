from abc import ABC, abstractmethod

import torch


class Method(torch.nn.Module, ABC):
    """ Abstract class for methods.
    """
    def __init__(self, **kwargs):
        super(Method, self).__init__()

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def before_weight_update(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

    def visualize(self, *args, **kwargs):
        return {}

    @property
    def device(self):
        params = list(self.parameters())
        if len(params) > 0:
            return params[0].device
        # This remaining case is for nn.DataParallel. The replicas of it have empty parameter lists
        for v in self.modules():
            if isinstance(v, torch.nn.Linear):
                return v.weight.device
        raise Exception("Cannot find device")

    @property
    def attributes_to_save(self):
        return dict()
