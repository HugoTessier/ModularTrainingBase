"""
These functions allow to take a ImageNet-like ResNet (default Pytorch implementation) or a CIFAR-like ResNet
(according to the implementation of Akamaster, but with the downsampling option set to 'B', i.e. with conv+bn, instead
of 'A', that is a padding function, because the latter is much more difficult to prune).

To prune a model, one can follow the following steps, according to this example:
    batchnorm_weight_magnitude(model)  # Computes the importance score of each BN to prune
    constrained_groups, free_layers = extract_resnet_groups(model)  # Split BNs into groups of constrained or free layers
    group_mean(constrained_groups)  # Defines the definitive importance score in the case of constrained layers
    programmable_masks_creator(constrained_groups, free_layers,
                               {'local': {'groups': [1],
                                          'pruning_rates': [0.2]},
                                'global': [{'groups': [0], 'pruning_rate': 0.5},
                                           {'groups': [2, 3], 'pruning_rate': 0.5}]})  # Creates masks for each group

    apply_mask(model)  # Applies the masks
    reduce_masked_network(model)  # Prunes the model

This design allows to easily fiddle with layers, groups, scores or mask before proceeding further, and also allows
to apply instead custom pruning criteria or pruning strategies for constrained layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


# Generic functions

def extract_resnet_groups(model: nn.Module) -> Tuple[List[Dict], Dict]:
    """
    Takes a CIFAR-like or ImageNet-like ResNet and separates its BatchNorm2d layers into different groups, depending on
    the dependencies that link these layers.
    For example, in a BasicBlock, the first BatchNorm2d layer is free to be pruned without constraint; however, the
    second one has to be pruned the same way as the input of the block (or its shortcut), and therefore belongs to a
    constrained group.
    Each 'stage' of a ResNet produce a different group, and the first stage also involves the very first BatchNorm2d
    layer of the network.

    :param model: A ResNet model.
    :return: A tuple of two elements. The first element is a list of dictionaries, each containing a set of layers that
    must be pruned the exact same way; the second element contains a dictionary, listing all the layers that are free
    to be pruned without constraints.
    """
    group_others = {}

    def extract_constrained_bns(block, key):
        group = {}
        for n, m in block.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                if key in n:
                    if 'downsample' in n or 'bn2' in n or 'shortcut' in n:
                        group[n] = m
                    else:
                        group_others[n] = m
        return group

    group_1 = extract_constrained_bns(model, 'layer1')
    group_1['bn1'] = model.bn1
    group_2 = extract_constrained_bns(model, 'layer2')
    group_3 = extract_constrained_bns(model, 'layer3')
    if hasattr(model, 'layer4'):  # ResNets for CIFAR have 3 stages instead of 4
        group_4 = extract_constrained_bns(model, 'layer4')
        return [group_1, group_2, group_3, group_4], group_others
    return [group_1, group_2, group_3], group_others


def programmable_masks_creator(constrained_groups: List[Dict],
                               free_layers: Dict,
                               pruning_scheme: Dict):
    """
    Generates the mask for each layer to prune in the network. The mask depends on both the importance metrics, that
    should already have been generated, and the distribution policy, that can be local or global. It is possible to
    tell which groups to prune locally or globally, and if two sets of groups have to be each pruned globally, but
    independently between each other.
    IMPORTANT: The masks are stored as a 'mask' attribute in the layers themselves. The function returns nothing.

    :param constrained_groups: List of dictionaries, containing the constrained groups of BatchNorm2d layers.
    :param free_layers: Dictionary containing the unconstrained layers.
    :param pruning_scheme: A dictionary describing the pruning scheme. Here are two examples:
        {'local': {'groups': [1,2,3,4],
                   'pruning_rates': [0.2,0.2,0.2,0.2]},
         'global': [{'groups': [0], 'pruning_rate': 0.5}]}
         Here, the constrained groups n°1, 2, 3 and 4 are pruned locally with the same pruning rate of 20%.
         The group of unconstrained layers, of arbitrary index '0', is pruned globally with a pruning rate of 50%.
         {'local': {'groups': [1],
                    'pruning_rates': [0.2]},
          'global': [{'groups': [0], 'pruning_rate': 0.5},
                     {'groups': [2,3], 'pruning_rate': 0.5}]}
         Here, the constrained group n°1 is pruned locally, with a pruning rate of 20%.
         The free layers are pruned globally, with a pruning rate of 50%.
         The groups n°2 and 3 are also pruned globally at 50%, but independently of the free layers.
         If the group n°4 exists, it is not pruned.
    """

    def local_mask(group, pruning_rate):
        for k, v in group.items():
            threshold = v.pruning_metric.sort()[0][int(len(v.pruning_metric) * pruning_rate)]
            mask = v.pruning_metric > threshold
            if torch.sum(mask) == 0:  # Prevents layer collapse
                mask[torch.argmax(v.pruning_metric)] = True
            setattr(v, 'mask', mask)

    def global_mask(groups, pruning_rate):
        metrics = []
        for g in groups:
            for k, v in g.items():
                metrics.append(v.pruning_metric)
        metrics = torch.cat(metrics).sort()[0]
        threshold = metrics[int(len(metrics) * pruning_rate)]
        for g in groups:
            for k, v in g.items():
                mask = v.pruning_metric > threshold
                if torch.sum(mask) == 0:  # Prevents layer collapse
                    mask[torch.argmax(v.pruning_metric)] = True
                setattr(v, 'mask', mask)

    if 'global' in pruning_scheme:
        for glob in pruning_scheme['global']:
            involved_groups = []
            for g in glob['groups']:
                if g == 0:
                    involved_groups.append(free_layers)
                else:
                    involved_groups.append(constrained_groups[g - 1])
            global_mask(involved_groups, glob['pruning_rate'])
    if 'local' in pruning_scheme:
        if len(pruning_scheme['local']['groups']) != len(pruning_scheme['local']['pruning_rates']):
            raise ValueError
        for g, r in zip(pruning_scheme['local']['groups'], pruning_scheme['local']['pruning_rates']):
            if g == 0:
                local_mask(free_layers, r)
            else:
                local_mask(constrained_groups[g - 1], r)


def apply_mask(model):
    """
    Applies the masks, that have been calculated, and applies them to the weights of the BatchNorm2d layers to prune.

    :param model: A ResNet model for which masks have already been computed.
    """
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            if hasattr(m, 'mask'):
                m.weight.data *= m.mask
                m.bias.data *= m.mask
                m.running_var.data *= m.mask
                m.running_mean.data *= m.mask
                delattr(m, 'mask')
            if hasattr(m, 'pruning_metric'):
                delattr(m, 'pruning_metric')


def reduce_masked_network(model: nn.Module):
    """
    Takes a model, whose weights have already been masked, and does some black magic to really reduce the architecture.
    The black magic involves detecting pruned input/output channels in every layer depending on whether their grad is 0
    or not after pruning. In order for the activations not to mess up this step, the ReLU functions are temporarily
    replaced with SiLU functions, whose derivative is not 0.
    The output model should be functional and have its parameters and operations reduced, with no inactive weights
    remaining.

    :param model: The model to reduce.
    :return: The reduced model.
    """

    def purge_model(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                mask = m.weight.grad.sum(dim=(1, 2, 3)) != 0
                m.weight.data = m.weight.data[mask, :, :, :]
                m.in_channels = mask.sum().item()
                mask = m.weight.grad.sum(dim=(0, 2, 3)) != 0
                m.weight.data = m.weight.data[:, mask, :, :]
                m.out_channels = mask.sum().item()
            if isinstance(m, torch.nn.BatchNorm2d):
                mask = m.weight.data != 0
                m.weight.data = m.weight.data[mask]
                m.bias.data = m.bias.data[mask]
                m.running_mean.data = m.running_mean.data[mask]
                m.running_var.data = m.running_var.data[mask]
                m.num_features = mask.sum().item()
                m.in_channels = mask.sum().item()
                m.out_channels = mask.sum().item()
            if isinstance(m, torch.nn.Linear):
                mask = m.weight.grad.sum(dim=(0,)) != 0
                m.weight.data = m.weight.data[:, mask]

    relu = F.relu
    setattr(F, 'relu', F.silu)
    model.zero_grad(set_to_none=True)
    output = model(torch.rand(1, 3, 224, 224).cuda())
    output.mean().backward()
    purge_model(model)
    model.zero_grad(set_to_none=True)
    setattr(F, 'relu', relu)
    return model


# These functions serve to define which metric to use in the case of layers that have to be pruned the same way.

def shortcuts_only(groups: List[Dict]):
    """
    Defines that, in each group in the list, all layers have to be pruned according to the importance metrics in the
    shortcut BatchNorm2d layer.
    IMPORTANT: This function redefines the 'pruning_metric' attribute of layers; it doesn't return anything.

    :param groups: Groups to prune.
    """
    for g in groups:
        metric = None
        for k, v in g.items():
            if 'downsample' in k or k == 'bn1' or 'shortcut' in k:
                metric = v.pruning_metric
        if metric is None:
            raise RuntimeError
        for k, v in g.items():
            v.pruning_metric = metric


def group_l1(groups: List[Dict]):
    """
    Defines that, in each group in the list, all layers have to be pruned according to the L1 norm of weights across
    all BatchNorm2d layers.
    IMPORTANT: This function redefines the 'pruning_metric' attribute of layers; it doesn't return anything.

    :param groups: Groups to prune.
    """
    for g in groups:
        metrics = []
        for k, v in g.items():
            metrics.append(v.pruning_metric)
        metrics = torch.stack(metrics)
        norm = torch.linalg.vector_norm(metrics, 1, 0)
        for k, v in g.items():
            v.pruning_metric = norm


def group_l2(groups: List[Dict]):
    """
    Defines that, in each group in the list, all layers have to be pruned according to the L2 norm of weights across
    all BatchNorm2d layers.
    IMPORTANT: This function redefines the 'pruning_metric' attribute of layers; it doesn't return anything.

    :param groups: Groups to prune.
    """
    for g in groups:
        metrics = []
        for k, v in g.items():
            metrics.append(v.pruning_metric)
        metrics = torch.stack(metrics)
        norm = torch.linalg.vector_norm(metrics, 2, 0)
        for k, v in g.items():
            v.pruning_metric = norm


def group_mean(groups: List[Dict]):
    """
    Defines that, in each group in the list, all layers have to be pruned according to the average of weights across
    all BatchNorm2d layers.
    IMPORTANT: This function redefines the 'pruning_metric' attribute of layers; it doesn't return anything.

    :param groups: Groups to prune.
    """
    for g in groups:
        metrics = []
        for k, v in g.items():
            metrics.append(v.pruning_metric)
        metrics = torch.stack(metrics)
        norm = torch.mean(metrics, dim=0)
        for k, v in g.items():
            v.pruning_metric = norm


# These functions are the pruning criteria themselves, that give the importance score of BatchNorm2d weights.


def batchnorm_weight_magnitude(model: nn.Module):
    """
    Defines the importance score of BatchNorm2d channels as the absolute magnitude of their weights.
    IMPORTANT: This function defines the importance score as a 'pruning_metric' attribute inside the layers themselves;
    it doesn't return anything.

    :param model: The model to prune.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            setattr(m, 'pruning_metric', m.weight.data.abs())
