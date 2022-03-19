from typing import Union, List, Tuple, Optional
from collections import defaultdict
import os
import pickle
import time
import copy
import logging

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim
import torch.utils.data
import numpy as np
import torch

from .callbacks import Callback, Stopper
from .metrics import Metric
from ..method_utils import Method
from .. import visualizations as vis
from .. import utils


def apply_weight_decay(named_params, weight_decay_rate):
    decay = []
    no_decay = []
    for name, param in named_params:
        if len(param.shape) == 1 or name.endswith(".bias"):  # BatchNorm1D or bias
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': weight_decay_rate}]


def build_optimizer(named_params, args):
    args = copy.deepcopy(args)  # as we are modifying it below

    # add weight decay if needed
    weight_decay_rate = args.pop('weight_decay', None)
    if weight_decay_rate is not None:
        params = apply_weight_decay(named_params, weight_decay_rate)
    else:
        params = [param for name, param in named_params]

    name = args.pop('name', 'adam')

    if name == 'adam':
        return utils.call_fn_ignoring_unexpected_args(torch.optim.Adam, params, **args)

    if name == 'sgd':
        return utils.call_fn_ignoring_unexpected_args(torch.optim.SGD, params, **args)

    raise ValueError(f"Optimizer with name '{name}' is not supported")


def build_scheduler(optimizer, args):
    name = args.get('name', 'StepLR')

    if name == 'StepLR':
        step_size = args.get('step_size', 1)
        gamma = args.get('gamma', 1.0)
        print(f"Using StepLR scheduler with step_size={step_size} and gamma={gamma}")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == 'MultiStepLR':
        milestones = args.get('milestones', [])
        gamma = args.get('gamma', 1.0)
        print(f"Using MultiStepLR scheduler with milestones={milestones} and gamma={gamma}")
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    raise ValueError(f"Scheduler with name '{name}' is not supported")


def make_markdown_table_from_dict(params_dict):
    table = "| param | value |  \n|:----|:-----|  \n"
    for k, v in params_dict.items():
        table += "| {} | {} |  \n".format(k, v)
    return table


class Trainer:
    """ Trainer takes care of setting up and running of the training process. """
    def __init__(self,
                 model: Method,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 n_epochs: int,
                 save_freq: int = 10,
                 vis_freq: int = 4,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 optimization_args: Optional[dict] = None,
                 log_dir: Optional[str] = None,
                 args_to_log=None,
                 metrics: Optional[Union[List[Metric], Tuple[Metric]]] = None,
                 callbacks: Optional[Union[List[Callback], Tuple[Callback]]] = None,
                 stopper: Optional[Stopper] = None,
                 device_ids: Optional[Union[list, tuple]] = None,
                 num_accumulation_steps: int = 1,
                 grad_clip_norm: Optional[Union[float, int]] = None):
        """
        :param model: The model to train.
        :param train_loader: Loader of getting the training data.
        :param val_loader: Loader for getting the validation data.
        :param n_epochs: The maximum number of training epochs.
        :param save_freq: Specifies the frequency of saving (measured in epochs).
        :param vis_freq: Specifies the frequency of calling visualizing (measured in epochs).
        :param optimizer: An optimizer instance. Note that it should be attached to the parameters of the model. If
                          this argument is None, an optimizer will be created using the optimization_args parameter.
        :param scheduler: A scheduler instance. Note that it should be attached to the correct optimizer. If this
                          argument is None, a scheduler will be created using the optimization_args parameter.
        :param optimization_args: A config dictionary specifying the optimizer and scheduler. If specified it should
                                  contain an 'optimizer' and 'scheduler' keys at the top level.
        :param log_dir: The directory in which tensorboard logs, checkpoints, and other information will be saved.
        :param args_to_log: A namespace of arguments to log in tensorboard and save in the log dir.
        :param metrics: A list of Metric instances used for evaluating the model during the training.
        :param callbacks: A list of Callback instances used for saving best models, logging, etc.
        :param stopper: A Stopper instance used for stopping the training.
        :param device_ids: list of device ids to be used during the training. If there are at least two devices,
                           distributed data training will be used via torch.nn.DataParallel. Note that PyTorch
                           requires and we rely on the fact that the first device should match with model.device.
        :param num_accumulation_steps: how many steps gradients should be averaged before updating the parameters.
        :param grad_clip_norm: used for clipping the gradients.

        :Assumptions:
            1. loaders return (batch_inputs, batch_labels), where both can be lists or torch.Tensors
        """

        self.model = model
        print(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.save_iter = save_freq
        self.vis_iter = vis_freq

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            assert optimization_args is not None, "If 'optimizer' is not given, 'optimization_args' should be given"
            assert 'optimizer' in optimization_args, "'optimization_args' must contain configs for optimizer"
            self.optimizer = build_optimizer(model.named_parameters(), optimization_args['optimizer'])

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            assert optimization_args is not None, "If 'scheduler' is not given, 'optimization_args' should be given"
            scheduler_args = optimization_args.get('scheduler', {})
            self.scheduler = build_scheduler(self.optimizer, scheduler_args)

        # if log_dir is not given, logging will be done a new directory in 'logs/' directory
        if log_dir is None:
            log_root = 'logs/'
            utils.make_path(log_root)
            last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
            self.log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))
            utils.make_path(self.log_dir)
        else:
            self.log_dir = log_dir

        self.tensorboard = SummaryWriter(self.log_dir)
        print("Visualize logs using: tensorboard --logdir={0}".format(self.log_dir))

        self.args_to_log = args_to_log

        if metrics is None:
            self.metrics = []
        else:
            self.metrics = metrics

        if callbacks is None:
            self.callbacks = []
        else:
            self.callbacks = callbacks

        self.stopper = stopper
        self.device_ids = device_ids
        self.num_accumulation_steps = num_accumulation_steps
        self.grad_clip_norm = grad_clip_norm

        self.data_parallel_model = None
        if (device_ids is not None) and len(device_ids) >= 2:
            print(f"Using multiple GPUs: {device_ids}")
            self.data_parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)

        self._update_iteration = 0  # this will be updated at each gradient update

    def _apply_weight_update(self):
        # some models might need to do something before applying gradients (e.g. adding noise)
        # TODO: if data parallelism is on, each model should call its before_weight_update
        self.model.before_weight_update()

        # log gradient norms
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            norm = utils.to_numpy(torch.norm(param.grad.detach()))
            self.tensorboard.add_scalar(f"gradient-norms/{name}", norm, self._update_iteration)

        self._update_iteration += 1

        # clip the gradients if needed
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm=self.grad_clip_norm)

        self.optimizer.step()

    def _run_partition(self, epoch, loader, partition, is_training):
        # Build the context dictionary. This will be passed to on_*_[start|end] callbacks.
        context = {
            'epoch': epoch,
            'loader': loader,
            'dataset': loader.dataset,
            'partition': partition,
            'is_training': is_training,
            'tensorboard': self.tensorboard,
            'trainer': self
        }

        # call on_partition_start callbacks
        self.model.on_partition_start(**context)
        for metric in self.metrics:
            metric.on_partition_start(**context)

        example_losses = defaultdict(list)
        current_step_idx = 0

        for (batch_data, batch_labels) in tqdm(loader, desc='{} batches'.format(partition)):
            # make the input and labels lists
            if isinstance(batch_data, torch.Tensor):
                batch_data = [batch_data]
            if isinstance(batch_labels, torch.Tensor):
                batch_labels = [batch_labels]

            # zero gradients in training phase
            if is_training and current_step_idx == 0:
                self.optimizer.zero_grad()

            # forward pass
            forward_model = (self.model if self.data_parallel_model is None else self.data_parallel_model)
            with torch.set_grad_enabled(is_training):
                outputs = forward_model.forward(inputs=batch_data, labels=batch_labels,
                                                grad_enabled=is_training, **context)

                batch_losses, outputs = self.model.compute_loss(inputs=batch_data, labels=batch_labels,
                                                                outputs=outputs, grad_enabled=is_training,
                                                                **context)
                batch_total_loss = sum([loss for name, loss in batch_losses.items()])

            if is_training:
                # backward pass
                batch_total_loss.backward()

                # update the parameters
                if current_step_idx == self.num_accumulation_steps - 1:
                    self._apply_weight_update()

            # call on_iteration_end callbacks
            self.model.on_iteration_end(outputs=outputs, batch_losses=batch_losses, batch_labels=batch_labels,
                                        **context)
            for metric in self.metrics:
                metric.on_iteration_end(outputs=outputs, batch_labels=batch_labels, **context)

            # collect all losses
            if len(batch_losses) > 1:
                batch_losses['total'] = batch_total_loss
            for k, v in batch_losses.items():
                example_losses['{}_{}'.format(partition, k)].append(utils.to_numpy(v))

            # update the step counter
            current_step_idx = (current_step_idx + 1) % self.num_accumulation_steps

        avg_losses = dict()
        for k, v in example_losses.items():
            avg_losses[k] = np.mean(v)
            self.tensorboard.add_scalar('losses/{}'.format(k), avg_losses[k], epoch)

        # if some gradient is left to apply
        if is_training and current_step_idx > 0:
            logging.warning('The number of training steps in one epoch is not a multiple of '
                            'number of accumulation steps')
            self._apply_weight_update()

        # call on_partition_end callbacks
        self.model.on_partition_end(**context)
        for metric in self.metrics:
            metric.on_partition_end(**context)

        return avg_losses

    def start(self):
        """ Starts the training. """

        # add args_to_log to tensorboard, but also store it separately for easier access
        if self.args_to_log is not None:
            self.tensorboard.add_text('script arguments table', make_markdown_table_from_dict(vars(self.args_to_log)))
            with open(os.path.join(self.log_dir, 'args.pkl'), 'wb') as f:
                pickle.dump(self.args_to_log, f)

        for epoch in range(self.n_epochs):
            t0 = time.time()

            self.model.train()
            if self.data_parallel_model is not None:
                self.data_parallel_model.train()
            train_losses = self._run_partition(epoch=epoch, loader=self.train_loader,
                                               partition='train', is_training=True)

            val_losses = {}
            if self.val_loader is not None:
                self.model.eval()
                if self.data_parallel_model is not None:
                    self.data_parallel_model.eval()
                val_losses = self._run_partition(epoch=epoch, loader=self.val_loader,
                                                 partition='val', is_training=False)

            # log some statistics
            t = time.time()
            log_string = 'Epoch: {}/{}'.format(epoch, self.n_epochs)
            for k, v in list(train_losses.items()) + list(val_losses.items()):
                log_string += ', {}: {:0.6f}'.format(k, v)
            log_string += ', Time: {:0.1f}s'.format(t - t0)
            print(log_string)

            # add visualizations
            if (epoch + 1) % self.vis_iter == 0:
                visualizations = self.model.visualize(self.train_loader, self.val_loader,
                                                      tensorboard=self.tensorboard, epoch=epoch)
                # visualizations is a dictionary containing figures in (name, fig) format.
                # there are visualizations created using matplotlib rather than tensorboard
                for (name, fig) in visualizations.items():
                    self.tensorboard.add_figure(name, fig, epoch)

            # save the model according to our schedule
            if (epoch + 1) % self.save_iter == 0:
                utils.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                           path=os.path.join(self.log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)))

            # call callbacks. These can be used to save the best model so far or initiate testing.
            for callback in self.callbacks:
                callback.on_epoch_end(epoch=epoch, model=self.model, optimizer=self.optimizer,
                                      scheduler=self.scheduler, log_dir=self.log_dir)

            # check whether the training should be ended
            if (self.stopper is not None) and self.stopper.is_stopping_time(epoch=epoch):
                print(f"Finishing the training at epoch {epoch}...")
                break

            # log the learning rate
            last_lr = self.scheduler.get_last_lr()
            if isinstance(last_lr, list):  # this happens when parameters are divided into groups
                last_lr = last_lr[0]
            self.tensorboard.add_scalar('hyper-parameters/lr', last_lr, epoch)

            # update the learning rate
            self.scheduler.step()

        # enable testing mode
        self.model.eval()

        # save the final version of the network
        utils.save(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                   path=os.path.join(self.log_dir, 'checkpoints', 'final.mdl'))

        # do final visualizations
        visualizations = self.model.visualize(self.train_loader, self.val_loader,
                                              tensorboard=self.tensorboard, epoch=self.n_epochs)
        for (name, fig) in visualizations.items():
            self.tensorboard.add_figure(name, fig, self.n_epochs)
            vis.savefig(fig, os.path.join(self.log_dir, name, 'final.png'))
