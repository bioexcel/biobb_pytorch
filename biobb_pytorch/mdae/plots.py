import os
from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle


def plot_loss(output_train_data_npz_path: str) -> None:
    """
    Plot the training and validation losses from the given npz file.

    Args:
        output_train_data_npz_path (str): The path to the npz file containing the training and validation losses.
    """
    npz_file = np.load(output_train_data_npz_path)
    train_loss = npz_file["train_loss"]
    val_loss = npz_file.get("valid_loss", None)
    min_train_loss_idx = np.argmin(train_loss)
    min_val_loss_idx = np.argmin(val_loss) if val_loss is not None else None

    plt.plot(
        range(len(train_loss)),
        train_loss,
        label=f"Training (min.: {min_train_loss_idx})",
        color="blue",
    )
    if val_loss is not None:
        plt.plot(
            range(len(val_loss)),
            val_loss,
            label=f"Validation (min.: {min_val_loss_idx})",
            color="orange",
        )
    plt.scatter(
        min_train_loss_idx,
        train_loss[min_train_loss_idx],
        color="blue",
        marker=MarkerStyle("v"),
        s=50,
    )
    if val_loss is not None and min_val_loss_idx is not None:
        plt.scatter(
            min_val_loss_idx,
            val_loss[min_val_loss_idx],
            color="orange",
            marker=MarkerStyle("v"),
            s=50,
        )
    plt.legend()
    plt.ylabel("Total Loss")
    plt.xlabel("Epochs")
    plt.title("Training/Validation")
    plt.show()


def plot_rmsd(input_xvg_path: Union[str, List[str]]) -> None:
    """
    Plots RMSD from one or more XVG files.

    Parameters:
    input_xvg_path (str or list of str): Path to a single XVG file or list of paths to multiple XVG files.

    The function parses each XVG file, extracts residue numbers and RMSD values,
    and plots them on a single figure for comparison.
    """
    if isinstance(input_xvg_path, str):
        input_xvg_path = [input_xvg_path]

    plt.figure(figsize=(15, 6))

    for file_path in input_xvg_path:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        # Load data from XVG, skipping comment lines
        data = np.loadtxt(file_path, comments=['#', '@'])

        # Assume column 0: time, column 1: RMSD
        time = data[:, 0]
        rmsd = data[:, 1]

        # Get label from filename
        label = os.path.basename(file_path).replace('.xvg', '')

        plt.plot(time, rmsd, label=label)

    plt.xlabel('time (ns)')
    plt.ylabel('RMSD (nm)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rmsf(input_xvg_path: Union[str, List[str]]) -> None:
    """
    Plots RMSF from one or more XVG files.

    Parameters:
    input_xvg_path (str or list of str): Path to a single XVG file or list of paths to multiple XVG files.

    The function parses each XVG file, extracts residue numbers and RMSF values,
    and plots them on a single figure for comparison.
    """
    if isinstance(input_xvg_path, str):
        input_xvg_path = [input_xvg_path]

    plt.figure(figsize=(15, 6))

    for file_path in input_xvg_path:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        # Load data from XVG, skipping comment lines
        data = np.loadtxt(file_path, comments=['#', '@'])

        # Assume column 0: residue, column 1: RMSF
        residues = data[:, 0]
        rmsf = data[:, 1]

        # Get label from filename
        label = os.path.basename(file_path).replace('.xvg', '')

        plt.plot(residues, rmsf, label=label)

    plt.xlabel('Residue Number')
    plt.ylabel('RMSF (nm)')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_rmsf_difference(input_xvg_path: Union[str, List[str]]) -> None:
    """
    Plots RMSF from one or more XVG files.

    Parameters:
    input_xvg_path (str or list of str): Path to a single XVG file or list of paths to multiple XVG files.

    The function parses each XVG file, extracts residue numbers and RMSF values,
    and plots them on a single figure for comparison.
    """
    if isinstance(input_xvg_path, str):
        input_xvg_path = [input_xvg_path]

    rmsfs = []
    for file_path in input_xvg_path:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        # Load data from XVG, skipping comment lines
        data = np.loadtxt(file_path, comments=['#', '@'])

        # Assume column 0: residue, column 1: RMSF
        residues = data[:, 0]
        rmsf = data[:, 1]

        # Get label from filename
        label = f"DIO: {os.path.basename(file_path).replace('xvg', '')} vs {os.path.basename(file_path).replace('xvg', '')}"

        rmsfs.append(rmsf)

    diff_rmsf = abs(rmsfs[0] - rmsfs[1])

    plt.figure(figsize=(10, 6))
    plt.plot(residues, diff_rmsf, label=label)
    plt.xlabel('Residue Number')
    plt.ylabel('RMSF (nm)')
    plt.title('RMSF Difference')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_latent_space(results_npz_path: str,
                      projection_dim: list,
                      snapshot_freq_ps=10) -> None:

    results = np.load(results_npz_path, allow_pickle=True)

    if 'z' not in results:
        raise KeyError(f"'z' not found in {results_npz_path}")

    z = results['z']

    if projection_dim is None:
        projection_dim = [0, 1]

    if len(projection_dim) != 2:
        raise ValueError(f"projection_dim must have length 2, got {projection_dim}")

    dim1, dim2 = projection_dim
    n_frames = z.shape[0]
    n_ticks = int(n_frames / 10)
    timestep_ns = 1 / snapshot_freq_ps

    plt.figure(figsize=(10, 6))
    plt.scatter(z[:, dim1], z[:, dim2], c=np.arange(n_frames) * timestep_ns, s=10, alpha=1.0)
    plt.xlabel(f'latent_dim {dim1}')
    plt.ylabel(f'latent_dim {dim2}')

    ticks = np.arange(n_frames)[::n_ticks] * timestep_ns
    if 0 not in ticks:
        ticks = np.insert(ticks, 0, 0)
    if (n_frames - 1) * timestep_ns not in ticks:
        ticks = np.append(ticks, (n_frames - 1) * timestep_ns)

    plt.colorbar(ticks=ticks, label='Time (ns)')
    plt.title('Latent Space Visualization')
    plt.show()
