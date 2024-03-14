""" Common functions for package biobb_pytorch.models """
import numpy as np
import torch
from typing import Callable


def ndarray_normalization(ndarray: np.ndarray, max_values: np.ndarray, min_values: np.ndarray) -> np.ndarray:
    """
    Normalize an ndarray along a specified axis.

    Args:
        ndarray (np.ndarray): The input ndarray to be normalized.
        max_values (np.ndarray): The maximum values for normalization.
        min_values (np.ndarray): The minimum values for normalization.

    Returns:
        np.ndarray: The normalized ndarray.
    """
    return (ndarray - min_values) / (max_values - min_values)


def ndarray_denormalization(normalized_ndarray: np.ndarray, max_values: np.ndarray, min_values: np.ndarray) -> np.ndarray:
    """
    Denormalizes a normalized ndarray using the given max and min values.

    Args:
        normalized_ndarray (np.ndarray): The normalized ndarray to be denormalized.
        max_values (np.ndarray): The maximum value used for normalization.
        min_values (np.ndarray): The minimum value used for normalization.

    Returns:
        np.ndarray: The denormalized ndarray.
    """
    return normalized_ndarray * (max_values - min_values) + min_values


def get_loss_function(loss_function: str) -> Callable:
    """
    Get the loss function from the given string.

    Args:
        loss_function (str): The loss function to be used.

    Returns:
        Callable: The loss function.
    """
    loss_function_dict = dict(filter(lambda pair: pair[0].endswith('Loss'), vars(torch.nn).items()))
    try:
        return loss_function_dict[loss_function]
    except KeyError:
        raise ValueError(f'Invalid loss function: {loss_function}')


def get_optimizer_function(optimizer_function: str) -> Callable:
    """
    Get the optimizer function from the given string.

    Args:
        optimizer_function (str): The optimizer function to be used.

    Returns:
        Callable: The optimizer function.
    """
    optimizer_function_dict = dict(filter(lambda pair: not pair[0].startswith('_'), vars(torch.optim).items()))
    try:
        return optimizer_function_dict[optimizer_function]
    except KeyError:
        raise ValueError(f'Invalid optimizer function: {optimizer_function}')
