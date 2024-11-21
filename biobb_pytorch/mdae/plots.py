import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.markers import MarkerStyle  # type: ignore


def plot_loss(output_train_data_npz_path: str) -> None:
    """
    Plot the training and validation losses from the given npz file.

    Args:
        output_train_data_npz_path (str): The path to the npz file containing the training and validation losses.
    """
    npz_file = np.load(output_train_data_npz_path)
    train_loss = npz_file["train_losses"]
    val_loss = npz_file["validation_losses"]
    min_train_loss_idx = np.argmin(train_loss)
    min_val_loss_idx = np.argmin(val_loss)
    plt.plot(
        range(len(train_loss)),
        train_loss,
        label=f"Training (min.: {min_train_loss_idx})",
        color="blue",
    )
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


def _numpy_rmsd(reference, trajectory):
    return np.sqrt(np.mean(np.sum((reference - trajectory) ** 2, axis=2), axis=1))


def plot_rmsd(traj_file_npy_path, output_reconstructed_traj_npy_path) -> None:
    perf_data = np.load(traj_file_npy_path)
    output = np.load(output_reconstructed_traj_npy_path)
    rmsd_trajectory = _numpy_rmsd(perf_data[0], perf_data) * 10  # Convert to Å
    rmsd_output = _numpy_rmsd(perf_data[0], output) * 10  # Convert to Å
    frames = np.arange(len(rmsd_trajectory))
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(frames, rmsd_trajectory, color="blue", linewidth=1, label="Original")
    ax.plot(frames, rmsd_output, color="red", linewidth=1, label="Reconstruction")
    # Labels, title, and legend
    ax.set_xlabel("# Frame")
    ax.set_ylabel("RMSD (Å)")
    plt.title("RMSD Plot")
    plt.legend()
    plt.show()


def plot_latent_space(latent_space_npy_path: str) -> None:
    z = np.load(latent_space_npy_path)
    gs = gridspec.GridSpec(4, 4)
    fig = plt.figure(figsize=(15, 10))
    ax_main = plt.subplot(gs[1:4, :3])
    ax_xDist = plt.subplot(gs[0, :3], sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:4, 3], sharey=ax_main)
    sc = ax_main.scatter(
        z[::1, 0], z[::1, 1], c=np.arange(len(z)), alpha=1, cmap="jet", s=2
    )
    # Position and size of colorbar based on ax_yDist
    pos = ax_yDist.get_position()
    cbar_ax = fig.add_axes((pos.x1 + 0.01, pos.y0, 0.02, pos.height))
    cbar = plt.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Frames")
    # X-axis marginal distribution
    ax_xDist.hist(z[::1, 0], bins=100, color="blue", alpha=0.7)
    # Y-axis marginal distribution
    ax_yDist.hist(
        z[::1, 1], bins=100, color="blue", alpha=0.7, orientation="horizontal"
    )
    ax_main.set_xlabel("z0", labelpad=20)
    ax_main.set_ylabel("z1", labelpad=20)
    plt.show()


def _numpy_rmsf_by_atom(trajectory):
    return np.sqrt(
        np.mean(np.sum((trajectory - np.mean(trajectory, axis=0)) ** 2, axis=2), axis=0)
    )


def plot_rmsf(original_traj_npy_file, mutated_reconstructed_traj_npy_file):
    original_traj = np.load(original_traj_npy_file)
    mutated_reconstructed_traj = np.load(mutated_reconstructed_traj_npy_file)
    rmsf_trajectory = _numpy_rmsf_by_atom(original_traj) * 10  # Convert to Å
    rmsf_output = _numpy_rmsf_by_atom(mutated_reconstructed_traj) * 10  # Convert to Å
    fig, ax = plt.subplots(figsize=(20, 6))
    indices = np.arange(len(rmsf_trajectory))
    ax.plot(indices, rmsf_trajectory, color="blue", linewidth=1, label="Original")
    ax.plot(indices, rmsf_output, color="red", linewidth=1, label="Reconstruction")
    ax.set_xlabel("# Atom")
    ax.set_ylabel("RMSD (Å) Average structure as reference")
    plt.title("RMSF Plot")
    plt.legend()
    plt.show()


def plot_rmsf_difference(original_traj_npy_file, mutated_reconstructed_traj_npy_file):
    original_traj = np.load(original_traj_npy_file)
    mutated_reconstructed_traj = np.load(mutated_reconstructed_traj_npy_file)
    rmsf_trajectory = _numpy_rmsf_by_atom(original_traj) * 10  # Convert to Å
    rmsf_output = _numpy_rmsf_by_atom(mutated_reconstructed_traj) * 10  # Convert to Å
    fig, ax = plt.subplots(figsize=(20, 6))
    indices = np.arange(len(rmsf_trajectory))
    # Plot RMSF for diference between input and output
    ax.plot(
        indices,
        (rmsf_trajectory - rmsf_output),
        color="orange",
        linewidth=1,
        label="DIO",
    )
    ax.axhline(y=1, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("# Atom")
    ax.set_ylabel("RMSD (Å)")
    plt.title("RMSF Plot")
    plt.legend()
    plt.show()
