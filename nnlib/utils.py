from torch.utils.data import Subset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import re
import inspect


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.device.type != 'cpu':
        x = x.cpu()
    return x.numpy()


def to_tensor(x, device='cpu', dtype=torch.float):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        x = torch.tensor(x, dtype=dtype, device=device)
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    return [to_cpu(xt) for xt in x]


def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value


def zero_grad(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param)


def save(model, path, optimizer=None, scheduler=None):
    print('Saving the model into {}'.format(path))
    make_path(os.path.dirname(path))

    save_dict = dict()
    save_dict['model'] = model.state_dict()
    save_dict['args'] = model.args

    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()

    torch.save(save_dict, path)


def load(path, methods, device=None, verbose=False, update_args_dict=None):
    print("Loading the model from {}".format(path))
    saved_dict = torch.load(path, map_location=device)
    args = saved_dict['args']
    if device is not None:
        args['device'] = device

    if update_args_dict is not None:
        for k, v in update_args_dict.items():
            args[k] = v

    model_class = getattr(methods, args['class'])
    model = model_class(**args)

    if verbose:
        print(model)

    model.load_state_dict(saved_dict['model'])
    model.eval()
    return model


def apply_on_dataset(model, dataset, batch_size=256, cpu=True, description="",
                     output_keys_regexp='.*', max_num_examples=2**30,
                     num_workers=0, **kwargs):
    model.eval()
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    outputs = defaultdict(list)

    for inputs_batch, labels_batch in tqdm(loader, desc=description):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        if not isinstance(labels_batch, list):
            labels_batch = [labels_batch]
        outputs_batch = model.forward(inputs=inputs_batch, labels=labels_batch, **kwargs)
        for k, v in outputs_batch.items():
            if re.fullmatch(output_keys_regexp, k) is None:
                continue
            if cpu:
                v = to_cpu(v)
            outputs[k].append(v)

        # add labels if requested
        if re.fullmatch(output_keys_regexp, 'labels') is not None:
            for label_idx, label_tensor in enumerate(labels_batch):
                outputs[f'label_{label_idx}'].append(label_tensor)

    for k in outputs:
        outputs[k] = torch.cat(outputs[k], dim=0)
        assert len(outputs[k]) == n_examples

    return outputs


def capture_arguments_of_init(init_func):
    def wrapper(self, *args, **kwargs):

        # get the name of kwargs, usually this will be just "kwargs"
        kwargs_name = inspect.getfullargspec(init_func).varkw

        # get the signature, bind arguments, apply defaults, and convert to dictionary
        signature = inspect.signature(init_func)
        bind_result = signature.bind(self, *args, **kwargs)
        bind_result.apply_defaults()
        argument_dict = bind_result.arguments

        # remove self
        argument_dict.pop('self')

        # remove kwargs and add its content to our dictionary
        kw = argument_dict.pop(kwargs_name)
        for k, v in kw.items():
            argument_dict[k] = v

        # call the init function
        init_func(self, *args, **kwargs)

        # add the class name
        argument_dict['class'] = self.__class__.__name__

        # write it in self
        self.args = argument_dict

    return wrapper
