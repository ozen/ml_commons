import os
import shutil
from collections import OrderedDict
import io
import torch
from tqdm import tqdm
import numpy as np


def save_checkpoint(checkpoint_dir, step, is_best, checkpoint_dict):
    if 'args' in checkpoint_dict:
        # TextIOWrappers are not serializable, convert them first
        args_ = {}
        for k, v in vars(checkpoint_dict['args']).items():
            if isinstance(v, io.TextIOWrapper):
                args_[k] = v.name
        vars(checkpoint_dict['args']).update(args_)

    path = os.path.join(checkpoint_dir, f'checkpoint_{step}.pt')
    torch.save(checkpoint_dict, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pt')
        shutil.copyfile(path, best_path)


def inference(model, loader, device='cpu'):
    try:
        from torch_geometric.data import Data, Batch
        has_to = (torch.Tensor, Data)
        geometric = True
    except ImportError:
        has_to = (torch.Tensor,)
        geometric = False

    def to_device(batch):
        if isinstance(batch, has_to):
            return batch.to(device)
        else:
            return tuple(elem.to(device) for elem in batch)

    def collate(output_list):
        if isinstance(output_list[0], torch.Tensor):
            return torch.cat(output_list, dim=0)
        elif geometric and isinstance(output_list[0], Data):
            return Batch.from_data_list(output_list)
        else:
            return [collate(dim) for dim in zip(*output_list)]

    with torch.no_grad():
        output = [model(to_device(batch)) for batch in tqdm(loader, 'Inference')]
    return collate(output)


def get_oversampler(dataset):
    num_classes = len(dataset.classes)
    count_per_class = np.histogram(dataset.targets, num_classes)[0]
    class_weights = np.max(count_per_class) / count_per_class
    instance_weights = np.repeat(class_weights, count_per_class)
    sampler = torch.utils.data.WeightedRandomSampler(instance_weights, 2048, replacement=True)
    return sampler


def subset_state_dict(state_dict, key):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if f'{key}.' in k:
            new_state_dict[k.replace(f'{key}.', '')] = v
    return new_state_dict


def weight_averaging(model_class, checkpoint_paths, data_loader, device):
    from torch.optim.swa_utils import AveragedModel, update_bn

    model = model_class.load_from_checkpoint(checkpoint_paths[0])
    swa_model = AveragedModel(model)

    for path in checkpoint_paths:
        model = model_class.load_from_checkpoint(path)
        swa_model.update_parameters(model)

    swa_model = swa_model.to(device)
    update_bn(data_loader, swa_model, device)
    return swa_model
