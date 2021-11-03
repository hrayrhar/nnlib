import torch
import torch.nn.functional as F
import torch.autograd

from nnlib.utils import capture_arguments_of_init
from nnlib import losses
from nnlib import nn_utils
from nnlib import visualizations as vis
from nnlib.method_utils import Method


class BaseClassifier(Method):
    """ Abstract class for classifiers.
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__(**kwargs)

    def visualize(self, train_loader, val_loader, **kwargs):
        visualizations = {}

        # visualize pred
        fig, _ = vis.plot_predictions(self, train_loader, key='pred')
        visualizations['predictions/pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='pred')
            visualizations['predictions/pred-val'] = fig

        return visualizations


class StandardClassifier(BaseClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args

        # create the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(device)

    def forward(self, inputs, labels=None, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)

        pred = self.classifier(x)

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        pred = outputs['pred']
        y = labels[0].to(self.device)

        # classification loss
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(target=y_one_hot, logits=pred,
                                                         loss_function='ce')

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, outputs
