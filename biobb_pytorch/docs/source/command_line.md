# BioBB PYTORCH Command Line Help
Generic usage:
```python
biobb_command [-h] --config CONFIG --input_file(s) <input_file(s)> --output_file <output_file>
```
-----------------


## Build_mdae
Build a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
build_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_stats_pt_path** (*string*): Path to the input model statistics file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt). Accepted formats: PT
* **output_model_pth_path** (*string*) (Optional): Optional path to save the built model (.pth). If omitted, the model is kept in memory only. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **model_type** (*string*): (AutoEncoder) Name of the model class to instantiate (must exist in biobb_pytorch.mdae.models).
* **n_cvs** (*integer*): (1) Dimensionality of the latent space.
* **encoder_layers** (*array*): ([16]) Number of neurons in each encoder layer.
* **decoder_layers** (*array*): ([16]) Number of neurons in each decoder layer.
* **options** (*object*): Additional options (e.g. norm_in, optimizer, loss_function, device, etc.).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_build_mdae.yml)
```python
properties:
  model_type: AutoEncoder
  n_cvs: 2
  encoder_layers: [16]
  decoder_layers: [16]
```
#### Command line
```python
build_mdae --config config_build_mdae.yml --input_stats_pt_path ref_input_model.pt --output_model_pth_path output_model.pth
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_build_mdae.json)
```python
{
  "properties": {
    "model_type": "AutoEncoder",
    "n_cvs": 2,
    "encoder_layers": [16],
    "decoder_layers": [16]
  }
}
```
#### Command line
```python
build_mdae --config config_build_mdae.json --input_stats_pt_path ref_input_model.pt --output_model_pth_path output_model.pth
```

## MDFeaturizer
Build Molecular Dynamics features for MDAE.
### Get help
Command:
```python
mdfeaturizer -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_trajectory_path** (*string*) (Optional): Path to the input trajectory file (optional, if omitted topology file is used as trajectory). File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.xtc). Accepted formats: XTC, DCD
* **input_topology_path** (*string*): Path to the input topology file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pdb). Accepted formats: PDB
* **input_labels_npy_path** (*string*) (Optional): Optional labels file in NumPy format. File type: input. Accepted formats: NPY
* **input_weights_npy_path** (*string*) (Optional): Optional weights file in NumPy format. File type: input. Accepted formats: NPY
* **output_dataset_pt_path** (*string*): Path to the output dataset file (.pt). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
* **output_stats_pt_path** (*string*): Path to the output statistics file (.pt). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **cartesian** (*object*): Options for Cartesian coordinates feature generation (e.g. selection, fit_selection).
* **distances** (*object*): Options for pairwise distance features (selection, cutoff, periodic, bonded, etc.).
* **angles** (*object*): Options for angle features (selection, periodic, bonded, etc.).
* **dihedrals** (*object*): Options for dihedral features (selection, periodic, bonded, etc.).
* **options** (*object*): General processing options (e.g. timelag, norm_in).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_md_feature_pipeline.yml)
```python
properties:
  cartesian:
    selection: 'name CA'
  distances:
    selection: 'name CA'
    cutoff: 0.4
    periodic: true
  options:
    timelag: 10
    norm_in:
      mode: 'min_max'
```
#### Command line
```python
mdfeaturizer --config config_md_feature_pipeline.yml --input_topology_path ref_output_model.pdb --output_dataset_pt_path ref_output_model.pt --output_stats_pt_path ref_output_model.pt
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_md_feature_pipeline.json)
```python
{
  "properties": {
    "cartesian": {
      "selection": "name CA"
    },
    "distances": {
      "selection": "name CA",
      "cutoff": 0.4,
      "periodic": true
    },
    "options": {
      "timelag": 10,
      "norm_in": {
        "mode": "min_max"
      }
    }
  }
}
```
#### Command line
```python
md_feature_pipeline --config config_md_feature_pipeline.json --input_topology_path ref_output_model.pdb --output_dataset_pt_path ref_output_model.pt --output_stats_pt_path ref_output_model.pt
```

## Train_mdae
Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
train_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the input (initial) model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) produced by the MD feature pipeline. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
* **output_model_pth_path** (*string*) (Optional): Optional path to save the trained model (.pth). If omitted, the trained model is only available in memory. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **output_metrics_npz_path** (*string*) (Optional): Optional path to save training metrics in compressed NumPy format (.npz). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_metrics.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Trainer** (*object*): PyTorch Lightning Trainer options (e.g. max_epochs, callbacks, logger, profiler, accelerator, devices, etc.).
* **Dataset** (*object*): mlcolvar DictDataset / DictModule options (e.g. batch_size, split proportions and shuffling flags).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_mdae.yml)
```python
properties:
  Trainer:
    max_epochs: 10
    callbacks:
      metrics: ['EarlyStopping']
  Dataset:
    batch_size: 32
    split:
      train_prop: 0.8
      val_prop: 0.2
```
#### Command line
```python
train_mdae --config config_train_mdae.yml --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_model_pth_path output_model.pth --output_metrics_npz_path ref_output_metrics.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_mdae.json)
```python
{
  "properties": {
    "Trainer": {
      "max_epochs": 10,
      "callbacks": {
        "metrics": ["EarlyStopping"]
      }
    },
    "Dataset": {
      "batch_size": 32,
      "split": {
        "train_prop": 0.8,
        "val_prop": 0.2
      }
    }
  }
}
```
#### Command line
```python
train_mdae --config config_train_mdae.json --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_model_pth_path output_model.pth --output_metrics_npz_path ref_output_metrics.npz
```

## Evaluate_mdae
Evaluate a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
evaluate_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) to evaluate on. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*): Path to the output evaluation results file (compressed NumPy archive). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): mlcolvar DictDataset / DataLoader options (e.g. batch_size, shuffle).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_evaluate_mdae.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false
```
#### Command line
```python
evaluate_mdae --config config_evaluate_mdae.yml --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_results.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_evaluate_mdae.json)
```python
{
  "properties": {
    "Dataset": {
      "batch_size": 32,
      "shuffle": false
    }
  }
}
```
#### Command line
```python
evaluate_mdae --config config_evaluate_mdae.json --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_results.npz
```

## Encode_mdae
Encode data with a Molecular Dynamics AutoEncoder (MDAE) model.
### Get help
Command:
```python
encode_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose encoder will be used. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) to encode. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*): Path to the output latent-space results file (compressed NumPy archive, typically containing 'z'). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_latent_space.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): mlcolvar DictDataset / DataLoader options (e.g. batch_size, shuffle).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_encode_mdae.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false
```
#### Command line
```python
encode_mdae --config config_encode_mdae.yml --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_latent_space.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_encode_mdae.json)
```python
{
  "properties": {
    "Dataset": {
      "batch_size": 32,
      "shuffle": false
    }
  }
}
```
#### Command line
```python
encode_mdae --config config_encode_mdae.json --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_latent_space.npz
```

## Decode_mdae
Decode latent variables with a Molecular Dynamics AutoEncoder (MDAE) model.
### Get help
Command:
```python
decode_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose decoder will be used. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_npy_path** (*string*): Path to the input latent variables file in NumPy format (e.g. encoded 'z'). File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_latent_space.npy). Accepted formats: NPY
* **output_results_npz_path** (*string*): Path to the output reconstructed data file (compressed NumPy archive, typically containing 'xhat'). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_reconstructed_data.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): DataLoader options (e.g. batch_size, shuffle) for batching the latent variables.
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_decode_mdae.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false
```
#### Command line
```python
decode_mdae --config config_decode_mdae.yml --input_model_pth_path output_model.pth --input_dataset_npy_path ref_input_latent_space.npy --output_results_npz_path ref_output_reconstructed_data.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_decode_mdae.json)
```python
{
  "properties": {
    "Dataset": {
      "batch_size": 32,
      "shuffle": false
    }
  }
}
```
#### Command line
```python
decode_mdae --config config_decode_mdae.json --input_model_pth_path output_model.pth --input_dataset_npy_path ref_input_latent_space.npy --output_results_npz_path ref_output_reconstructed_data.npz
```

## Generate_plumed_mdae
Generate PLUMED input for biased dynamics using an MDAE model.
### Get help
Command:
```python
generate_plumed_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained PyTorch model (.pth) to be converted to TorchScript and used in PLUMED. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_stats_pt_path** (*string*) (Optional): Optional statistics file (.pt) produced during featurization, used to derive the PLUMED features.dat content. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt). Accepted formats: PT
* **input_reference_pdb_path** (*string*) (Optional): Optional reference PDB used for FIT_TO_TEMPLATE actions when Cartesian features are present. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pdb). Accepted formats: PDB
* **input_ndx_path** (*string*) (Optional): Optional GROMACS index (NDX) file used to define groups when required by PLUMED. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_index.ndx). Accepted formats: NDX
* **output_plumed_dat_path** (*string*): Path to the output PLUMED input file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_plumed.dat). Accepted formats: DAT
* **output_features_dat_path** (*string*): Path to the output features.dat file describing the CVs to PLUMED. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_features.dat). Accepted formats: DAT
* **output_model_ptc_path** (*string*): Path to the output TorchScript model file (.ptc) for PLUMED's PYTORCH_MODEL action. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.ptc). Accepted formats: PTC
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **additional_actions** (*array*): Additional PLUMED actions to prepend (e.g. ENERGY, RMSD).
* **group** (*object*): GROUP definition options (label, NDX group or atom selection parameters).
* **wholemolecules** (*object*): WHOLEMOLECULES options when using Cartesian coordinates.
* **fit_to_template** (*object*): FIT_TO_TEMPLATE options (e.g. STRIDE, TYPE, etc.).
* **pytorch_model** (*object*): PYTORCH_MODEL options (label, PACE and other parameters).
* **bias** (*array*): List of biasing actions (e.g. METAD) to be added to the PLUMED file.
* **prints** (*object*): PRINT command parameters (e.g. ARG, STRIDE, FILE).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_generate_plumed_mdae.yml)
```python
properties:
  additional_actions:
    - name: ENERGY
      label: ene
  group:
    label: c_alphas
    NDX_GROUP: chA_&_C-alpha
  wholemolecules:
    ENTITY0: c_alphas
  fit_to_template:
    STRIDE: 1
    TYPE: OPTIMAL
  pytorch_model:
    label: cv
    PACE: 1
  bias:
    - name: METAD
      label: bias
      params:
        ARG: cv.1
        PACE: 500
        HEIGHT: 1.2
        SIGMA: 0.35
        FILE: HILLS
        BIASFACTOR: 8
  prints:
    ARG: cv.*,bias.*
    STRIDE: 1
    FILE: COLVAR
```
#### Command line
```python
generate_plumed_mdae --config config_generate_plumed_mdae.yml --input_model_pth_path output_model.pth --input_stats_pt_path ref_input_model.pt --output_plumed_dat_path output_plumed.dat --output_features_dat_path output_features.dat --output_model_ptc_path output_model.ptc
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_generate_plumed_mdae.json)
```python
{
  "properties": {
    "additional_actions": [
      {
        "name": "ENERGY",
        "label": "ene"
      }
    ],
    "group": {
      "label": "c_alphas",
      "NDX_GROUP": "chA_&_C-alpha"
    },
    "wholemolecules": {
      "ENTITY0": "c_alphas"
    },
    "fit_to_template": {
      "STRIDE": 1,
      "TYPE": "OPTIMAL"
    },
    "pytorch_model": {
      "label": "cv",
      "PACE": 1
    },
    "bias": [
      {
        "name": "METAD",
        "label": "bias",
        "params": {
          "ARG": "cv.1",
          "PACE": 500,
          "HEIGHT": 1.2,
          "SIGMA": 0.35,
          "FILE": "HILLS",
          "BIASFACTOR": 8
        }
      }
    ],
    "prints": {
      "ARG": "cv.*,bias.*",
      "STRIDE": 1,
      "FILE": "COLVAR"
    }
  }
}
```
#### Command line
```python
generate_plumed_mdae --config config_generate_plumed_mdae.json --input_model_pth_path output_model.pth --input_stats_pt_path ref_input_model.pt --output_plumed_dat_path output_plumed.dat --output_features_dat_path output_features.dat --output_model_ptc_path output_model.ptc
```

## Feat2traj_mdae
Convert reconstructed features to a trajectory.
### Get help
Command:
```python
feat2traj_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_results_npz_path** (*string*): Path to the input reconstructed results file (.npz), typically containing an 'xhat' array. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_results.npz). Accepted formats: NPZ
* **input_stats_pt_path** (*string*): Path to the input model statistics file (.pt) containing cartesian indices and optionally topology. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt). Accepted formats: PT
* **input_topology_path** (*string*) (Optional): Optional topology file (PDB) used if no suitable topology is found in the stats file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pdb). Accepted formats: PDB
* **output_traj_path** (*string*): Path to the output trajectory file (xtc, dcd or pdb). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.xtc). Accepted formats: XTC, DCD, PDB
* **output_top_path** (*string*) (Optional): Optional PDB topology file to write when the trajectory format requires a separate topology (e.g. xtc or dcd). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pdb). Accepted formats: PDB
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* No additional properties are currently used.
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_feat2traj_mdae.yml)
```python
properties: {}
```
#### Command line
```python
feat2traj_mdae --config config_feat2traj_mdae.yml --input_results_npz_path ref_input_results.npz --input_stats_pt_path ref_input_model.pt --output_traj_path output_model.xtc --output_top_path output_model.pdb
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_feat2traj_mdae.json)
```python
{
  "properties": {}
}
```
#### Command line
```python
feat2traj_mdae --config config_feat2traj_mdae.json --input_results_npz_path ref_input_results.npz --input_stats_pt_path ref_input_model.pt --output_traj_path output_model.xtc --output_top_path output_model.pdb
```

## Lrp_mdae
Layer-wise Relevance Propagation (LRP) for an MDAE encoder.
### Get help
Command:
```python
lrp_mdae -h
```
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose encoder is analyzed. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) used for computing relevance scores. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*) (Optional): Optional path to the output results file containing relevance scores (compressed NumPy archive). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_lrp_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): Dataset/DataLoader options (e.g. batch_size and optional indices to subset the dataset).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_lrp_mdae.yml)
```python
properties:
  Dataset:
    batch_size: 32
    indices: [0, 1, 2, 3, 4]
```
#### Command line
```python
lrp_mdae --config config_lrp_mdae.yml --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_lrp_results.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_lrp_mdae.json)
```python
{
  "properties": {
    "Dataset": {
      "batch_size": 32,
      "indices": [0, 1, 2, 3, 4]
    }
  }
}
```
#### Command line
```python
lrp_mdae --config config_lrp_mdae.json --input_model_pth_path output_model.pth --input_dataset_pt_path ref_output_model.pt --output_results_npz_path ref_output_lrp_results.npz
```
