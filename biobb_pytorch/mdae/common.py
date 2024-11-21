"""Common functions for package biobb_pytorch.models"""

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch


def ndarray_normalization(
    ndarray: np.ndarray, max_values: np.ndarray, min_values: np.ndarray
) -> np.ndarray:
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


def ndarray_denormalization(
    normalized_ndarray: np.ndarray, max_values: np.ndarray, min_values: np.ndarray
) -> np.ndarray:
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
    loss_function_dict = dict(
        filter(lambda pair: pair[0].endswith("Loss"), vars(torch.nn).items())
    )
    try:
        return loss_function_dict[loss_function]
    except KeyError:
        raise ValueError(f"Invalid loss function: {loss_function}")


def get_optimizer_function(optimizer_function: str) -> Callable:
    """
    Get the optimizer function from the given string.

    Args:
        optimizer_function (str): The optimizer function to be used.

    Returns:
        Callable: The optimizer function.
    """
    optimizer_function_dict = dict(
        filter(lambda pair: not pair[0].startswith("_"), vars(torch.optim).items())
    )
    try:
        return optimizer_function_dict[optimizer_function]
    except KeyError:
        raise ValueError(f"Invalid optimizer function: {optimizer_function}")


def execute_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    input_dimensions: int,
    latent_dimensions: int,
    loss_function: Optional[torch.nn.modules.loss._Loss] = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    z_list: list[float] = []
    x_hat_list: list[float] = []
    with torch.no_grad():
        for data in dataloader:
            data = data[0].to(model.device)
            latent, output = model(data)
            if loss_function:
                loss = loss_function(output, data)
                losses.append(loss.item())
            z_list.append(latent.cpu().numpy())
            x_hat_list.append(output.cpu().numpy())
    loss = float(np.mean(losses)) if losses else -1.0
    latent_space: np.ndarray = np.reshape(
        np.concatenate(z_list, axis=0), (-1, latent_dimensions)
    )
    reconstructed_data: np.ndarray = np.reshape(
        np.concatenate(x_hat_list, axis=0), (-1, input_dimensions)
    )
    return loss, latent_space, reconstructed_data


def format_time(seconds: Union[float, int]) -> str:
    """Converts time in seconds to a string of the format 'HH:MM:SS'."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{:02}h {:02}m {:02}s".format(int(hours), int(minutes), int(seconds))
    elif minutes:
        return "{:02}m {:02}s".format(int(minutes), int(seconds))
    else:
        return "{:02}s".format(int(seconds))


def human_readable_file_size(file_path: Union[str, Path]) -> str:
    """Get the size of a file and return it in a human-readable format."""
    file_path = Path(file_path)  # Ensure file_path is a Path object
    size_in_bytes: float = file_path.stat().st_size
    units = ["Bytes", "KB", "MB", "GB", "PB"]
    for unit in units:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} {unit}"
