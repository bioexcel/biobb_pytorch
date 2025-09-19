import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import os
import mdtraj as md

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

def _numpy_rmsf_by_atom(traj: np.ndarray) -> np.ndarray:
    """Compute RMSF by atom from a trajectory numpy array."""
    mean_structure = np.mean(traj, axis=0)
    diff = traj - mean_structure
    squared_diff = np.square(diff)
    mean_squared_diff = np.mean(squared_diff, axis=0)
    rmsf = np.sqrt(np.sum(mean_squared_diff, axis=1))
    return rmsf


def plot_rmsf_difference(reference_traj_npy_file, mutated_reconstructed_traj_npy_file):
    original_traj = np.load(reference_traj_npy_file)
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


def plot_rmsd(input_dataset_pt_path: str,
              input_stats_pt_path: str,
              input_topology_path: str,
              output_model_results_file: str | list[str] | None = None,
             ):
    """
    Plots RMSF of the original dataset and reconstructed trajectories if model results are provided.
    
    Args:
        input_dataset_pt_path (str): Path to the dataset .pt file.
        input_stats_pt_path (str): Path to the stats .pt file.
        input_topology_path (str): Path to the topology file (e.g., PDB) for mdtraj.
        output_model_results_file (str | list[str] | None, optional): Path(s) to the model results .npz file(s) containing 'xhat'. Defaults to None.
        per_residue (bool, optional): If True, average RMSF per residue. Defaults to False.
    
    Raises:
        ValueError: If required data is missing or shapes mismatch.
    """
    # Load dataset and stats
    dataset = torch.load(input_dataset_pt_path, weights_only=False)
    stats = torch.load(input_stats_pt_path, weights_only=False)
    
    if 'cartesian_indices' not in stats:
        raise ValueError("No 'cartesian_indices' found in stats; cannot compute cartesian coordinates.")
    
    cartesian_range = len(stats['cartesian_indices'])
    cartesian = dataset['data'][:, :cartesian_range * 3].reshape(-1, cartesian_range, 3)
    
    # Load topology
    top = md.load_topology(input_topology_path)
    
    # Create original trajectory
    orig_traj = md.Trajectory(cartesian, top)
    
    # Compute RMSF for original (aligns to first frame internally)
    rmsd_orig = md.rmsd(orig_traj, orig_traj, frame=0)
    
    rmsds = [rmsd_orig]
    labels = ['Original']
    
    x_label = 'Number of Frames'
    x = np.arange(len(rmsd_orig))
    
    # If results files provided, compute for each reconstruction
    if output_model_results_file is not None:
        if isinstance(output_model_results_file, str):
            results_files = [output_model_results_file]
        else:
            results_files = output_model_results_file
        
        for file_path in results_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Results file not found: {file_path}")
            
            results = np.load(file_path, allow_pickle=True)
            
            if 'xhat' not in results:
                raise KeyError(f"'xhat' not found in {file_path}")
            
            xhat = results['xhat']
            recon_cartesian = xhat[:, :cartesian_range * 3].reshape(-1, cartesian_range, 3)
            
            if recon_cartesian.shape != cartesian.shape:
                raise ValueError(f"Shape mismatch between original and reconstructed cartesian in {file_path}: {cartesian.shape} vs {recon_cartesian.shape}")
            
            recon_traj = md.Trajectory(recon_cartesian, top)
            rmsd_recon = md.rmsd(recon_traj, recon_traj, frame=0)
            
            rmsds.append(rmsd_recon)
            
            labels.append(os.path.basename(file_path))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for rmsd, label in zip(rmsds, labels):
        plt.plot(x, rmsd, label=label)
    
    plt.xlabel(x_label)
    plt.ylabel('RMSD (nm)')
    plt.title('Root Mean Square Fluctuation')
    if len(rmsds) > 1:
        plt.legend()
    plt.show()

def plot_rmsf(input_dataset_pt_path: str,
              input_stats_pt_path: str,
              input_topology_path: str,
              output_model_results_file: str | list[str] | None = None,
              per_residue: bool = False):
    """
    Plots RMSF of the original dataset and reconstructed trajectories if model results are provided.
    
    Args:
        input_dataset_pt_path (str): Path to the dataset .pt file.
        input_stats_pt_path (str): Path to the stats .pt file.
        input_topology_path (str): Path to the topology file (e.g., PDB) for mdtraj.
        output_model_results_file (str | list[str] | None, optional): Path(s) to the model results .npz file(s) containing 'xhat'. Defaults to None.
        per_residue (bool, optional): If True, average RMSF per residue. Defaults to False.
    
    Raises:
        ValueError: If required data is missing or shapes mismatch.
    """
    # Load dataset and stats
    dataset = torch.load(input_dataset_pt_path, weights_only=False)
    stats = torch.load(input_stats_pt_path, weights_only=False)
    
    if 'cartesian_indices' not in stats:
        raise ValueError("No 'cartesian_indices' found in stats; cannot compute cartesian coordinates.")
    
    cartesian_range = len(stats['cartesian_indices'])
    cartesian = dataset['data'][:, :cartesian_range * 3].reshape(-1, cartesian_range, 3)
    
    # Load topology
    top = md.load_topology(input_topology_path)
    
    # Create original trajectory
    orig_traj = md.Trajectory(cartesian, top)
    
    # Compute RMSF for original (aligns to first frame internally)
    rmsf_orig = md.rmsf(orig_traj, orig_traj, frame=0)
    
    rmsfs = [rmsf_orig]
    labels = ['Original']
    
    # Process per residue if requested
    if per_residue:
        residues = np.array([atom.residue.resSeq for atom in top.atoms])
        unique_res = np.unique(residues)
        rmsf_orig_per_res = np.array([np.mean(rmsf_orig[residues == res]) for res in unique_res])
        rmsfs = [rmsf_orig_per_res]
        x_label = 'Residue Index'
        x = unique_res
    else:
        x_label = 'Atom Index'
        x = np.arange(len(rmsf_orig))
    
    # If results files provided, compute for each reconstruction
    if output_model_results_file is not None:
        if isinstance(output_model_results_file, str):
            results_files = [output_model_results_file]
        else:
            results_files = output_model_results_file
        
        for file_path in results_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Results file not found: {file_path}")
            
            results = np.load(file_path, allow_pickle=True)
            
            if 'xhat' not in results:
                raise KeyError(f"'xhat' not found in {file_path}")
            
            xhat = results['xhat']
            recon_cartesian = xhat[:, :cartesian_range * 3].reshape(-1, cartesian_range, 3)
            
            if recon_cartesian.shape != cartesian.shape:
                raise ValueError(f"Shape mismatch between original and reconstructed cartesian in {file_path}: {cartesian.shape} vs {recon_cartesian.shape}")
            
            recon_traj = md.Trajectory(recon_cartesian, top)
            rmsf_recon = md.rmsf(recon_traj, recon_traj, frame=0)
            
            if per_residue:
                rmsf_recon_per_res = np.array([np.mean(rmsf_recon[residues == res]) for res in unique_res])
                rmsfs.append(rmsf_recon_per_res)
            else:
                rmsfs.append(rmsf_recon)
            
            labels.append(os.path.basename(file_path))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for rmsf, label in zip(rmsfs, labels):
        plt.plot(x, rmsf, label=label)
    
    plt.xlabel(x_label)
    plt.ylabel('RMSF (nm)')
    plt.title('Root Mean Square Fluctuation')
    if len(rmsfs) > 1:
        plt.legend()
    plt.show()

def plot_latent_space(output_model_results_file: str | list[str],
                      projection_dim: int = 2):
    """
    Plots the latent space from model results. Projects to 2D or 3D if necessary using PCA.
    For multiple files, plots on the same figure with different colors.
    
    Args:
        output_model_results_file (str | list[str]): Path(s) to the model results .npz file(s) containing 'z'.
    
    Raises:
        ValueError: If projection_dim not 2 or 3, or if 'z' missing, or dims mismatch across files.
    """
    
    # Standardize to list
    if isinstance(output_model_results_file, str):
        results_files = [output_model_results_file]
    else:
        results_files = output_model_results_file
    
    zs_projected = []
    labels = []
    latent_dim = None
    num_frames = None
    
    for file_path in results_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        results = np.load(file_path, allow_pickle=True)
        
        if 'z' not in results:
            raise KeyError(f"'z' not found in {file_path}")
        
        z = results['z']
        projection_dim = int(z.shape[1])

        if projection_dim not in (2, 3):
            projection_dim = 2  
        
        if not isinstance(z, np.ndarray):
            z = np.array(z)
        
        if latent_dim is None:
            latent_dim = z.shape[1]
        elif z.shape[1] != latent_dim:
            raise ValueError(f"Latent dimension mismatch in {file_path}: expected {latent_dim}, got {z.shape[1]}")
        
        if num_frames is None:
            num_frames = z.shape[0]
        elif z.shape[0] != num_frames:
            raise ValueError(f"Number of frames mismatch in {file_path}: expected {num_frames}, got {z.shape[0]}")
        
        # Project if necessary
        z_proj = z[:, :projection_dim]  # Take first dims if <=
        
        zs_projected.append(z_proj)
        labels.append(os.path.basename(file_path))
    
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    if projection_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Latent Dim 3 (or PC3)')
    else:
        ax = fig.add_subplot(111)
    
    for z_proj, label in zip(zs_projected, labels):
        frames = np.arange(z_proj.shape[0])
        if projection_dim == 2:
            sc = ax.scatter(z_proj[:, 0], z_proj[:, 1], c=frames, cmap='viridis', label=label)
        elif projection_dim == 3:
            sc = ax.scatter(z_proj[:, 0], z_proj[:, 1], z_proj[:, 2], c=frames, cmap='viridis', label=label)
    
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_title('Latent Space Projection')
    if len(zs_projected) > 1:
        plt.legend()
    fig.colorbar(sc, ax=ax, label='Frame')
    plt.show()