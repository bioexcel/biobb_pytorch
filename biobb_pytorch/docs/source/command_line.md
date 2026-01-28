# BioBB PYTORCH Command Line Help
Generic usage:
```python
biobb_command [-h] --config CONFIG --input_file(s) <input_file(s)> --output_file <output_file>
```
-----------------


## Build_model
Build a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
buildModel -h
```
    usage: buildModel [-h] [-c CONFIG] -i INPUT_STATS_PT_PATH [-o OUTPUT_MODEL_PTH_PATH]
    
    Build a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      -i INPUT_STATS_PT_PATH, --input_stats_pt_path INPUT_STATS_PT_PATH
                            Path to the input model statistics file. Accepted formats: pt.
    
    optional arguments:
      -o OUTPUT_MODEL_PTH_PATH, --output_model_pth_path OUTPUT_MODEL_PTH_PATH
                            Path to save the model in .pth format. Accepted formats: pth.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_stats_pt_path** (*string*): Path to the input model statistics file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt). Accepted formats: PT
* **output_model_pth_path** (*string*): Path to save the model in .pth format. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **model_type** (*string*): (AutoEncoder) Name of the model class to instantiate (must exist in biobb_pytorch.mdae.models).
* **n_cvs** (*integer*): (1) Dimensionality of the latent space.
* **encoder_layers** (*array*): ([16]) List of integers representing the number of neurons in each encoder layer.
* **decoder_layers** (*array*): ([16]) List of integers representing the number of neurons in each decoder layer.
* **options** (*object*): ({'norm_in': {'mode': 'min_max'}}) Additional options (e.g. norm_in, optimizer, loss_function, device, etc.).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_build_model.yml)
```python
properties:
  decoder_layers:
  - 16
  encoder_layers:
  - 16
  model_type: AutoEncoder
  n_cvs: 2

```
#### Command line
```python
buildModel --config config_build_model.yml --input_stats_pt_path ref_input_model.pt --output_model_pth_path output_model.pth
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_build_model.json)
```python
{
  "properties": {
    "model_type": "AutoEncoder",
    "n_cvs": 2,
    "encoder_layers": [
      16
    ],
    "decoder_layers": [
      16
    ]
  }
}
```
#### Command line
```python
buildModel --config config_build_model.json --input_stats_pt_path ref_input_model.pt --output_model_pth_path output_model.pth
```

## evaluate_decoder
Evaluates a PyTorch autoencoder Decoder from the given properties.
### Get help
Command:
```python
evaluateDecoder -h
```
    usage: evaluateDecoder [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH --input_dataset_npy_path INPUT_DATASET_NPY_PATH -o OUTPUT_RESULTS_NPZ_PATH
    
    Evaluates a PyTorch autoencoder from the given properties.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the trained model file whose decoder will be used. Accepted formats: pth.
      --input_dataset_npy_path INPUT_DATASET_NPY_PATH
                            Path to the input latent variables file in NumPy format (e.g. encoded 'z'). Accepted formats: npy.
      -o OUTPUT_RESULTS_NPZ_PATH, --output_results_npz_path OUTPUT_RESULTS_NPZ_PATH
                            Path to the output reconstructed data file (compressed NumPy archive, typically containing 'xhat'). Accepted formats: npz.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose decoder will be used. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_npy_path** (*string*): Path to the input latent variables file in NumPy format (e.g. encoded 'z'). File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.npy). Accepted formats: NPY
* **output_results_npz_path** (*string*): Path to the output reconstructed data file (compressed NumPy archive, typically containing 'xhat'). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): ({}) DataLoader options (e.g. batch_size, shuffle) for batching the latent variables.
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_decode_model.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false

```
#### Command line
```python
evaluateDecoder --config config_decode_model.yml --input_model_pth_path output_model.pth --input_dataset_npy_path output_model.npy --output_results_npz_path output_results.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_decode_model.json)
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
evaluateDecoder --config config_decode_model.json --input_model_pth_path output_model.pth --input_dataset_npy_path output_model.npy --output_results_npz_path output_results.npz
```

## Evaluate_encoder
Encode data with a Molecular Dynamics AutoEncoder (MDAE) model.
### Get help
Command:
```python
evaluateEncoder -h
```
    usage: evaluateEncoder [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH --input_dataset_pt_path INPUT_DATASET_PT_PATH -o OUTPUT_RESULTS_NPZ_PATH
    
    Encode data with a Molecular Dynamics AutoEncoder (MDAE) model.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the trained model file whose encoder will be used. Accepted formats: pth.
      --input_dataset_pt_path INPUT_DATASET_PT_PATH
                            Path to the input dataset file (.pt) to encode. Accepted formats: pt.
      -o OUTPUT_RESULTS_NPZ_PATH, --output_results_npz_path OUTPUT_RESULTS_NPZ_PATH
                            Path to the output latent-space results file (compressed NumPy archive, typically containing 'z'). Accepted formats: npz.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose encoder will be used. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) to encode. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*): Path to the output latent-space results file (compressed NumPy archive, typically containing 'z'). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): ({}) mlcolvar DictDataset / DataLoader options (e.g. batch_size, shuffle).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_encode_model.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false

```
#### Command line
```python
evaluateEncoder --config config_encode_model.yml --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_results_npz_path output_results.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_encode_model.json)
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
evaluateEncoder --config config_encode_model.json --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_results_npz_path output_results.npz
```

## Evaluate_model
Evaluate a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
evaluateModel -h
```
    usage: evaluateModel [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH --input_dataset_pt_path INPUT_DATASET_PT_PATH -o OUTPUT_RESULTS_NPZ_PATH
    
    Evaluate a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the trained model file. Accepted formats: pth.
      --input_dataset_pt_path INPUT_DATASET_PT_PATH
                            Path to the input dataset file (.pt) to evaluate on. Accepted formats: pt.
      -o OUTPUT_RESULTS_NPZ_PATH, --output_results_npz_path OUTPUT_RESULTS_NPZ_PATH
                            Path to the output evaluation results file (compressed NumPy archive). Accepted formats: npz.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) to evaluate on. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*): Path to the output evaluation results file (compressed NumPy archive). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): ({}) mlcolvar DictDataset / DataLoader options (e.g. batch_size, shuffle).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_evaluate_model.yml)
```python
properties:
  Dataset:
    batch_size: 32
    shuffle: false

```
#### Command line
```python
evaluateModel --config config_evaluate_model.yml --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_results_npz_path output_results.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_evaluate_model.json)
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
evaluateModel --config config_evaluate_model.json --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_results_npz_path output_results.npz
```

## Feat2traj
Converts a .pt file (features) to a trajectory using cartesian indices and topology from the stats file.
### Get help
Command:
```python
feat2traj -h
```
    usage: feat2traj [-h] [-c CONFIG] --input_results_npz_path INPUT_RESULTS_NPZ_PATH --input_stats_pt_path INPUT_STATS_PT_PATH [--input_topology_path INPUT_TOPOLOGY_PATH] --output_traj_path OUTPUT_TRAJ_PATH [--output_top_path OUTPUT_TOP_PATH]
    
    Converts a .pt file (features) to a trajectory using cartesian indices and topology from the stats file.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_results_npz_path INPUT_RESULTS_NPZ_PATH
                            Path to the input reconstructed results file (.npz), typically containing an 'xhat' array. Accepted formats: npz.
      --input_stats_pt_path INPUT_STATS_PT_PATH
                            Path to the input model statistics file (.pt) containing cartesian indices and optionally topology. Accepted formats: pt.
      --output_traj_path OUTPUT_TRAJ_PATH
                            Path to save the trajectory in xtc/pdb/dcd format. Accepted formats: xtc, pdb, dcd.
    
    optional arguments:
      --input_topology_path INPUT_TOPOLOGY_PATH
                            Path to the topology file (PDB) used if no suitable topology is found in the stats file. Used if no topology is found in stats. Accepted formats: pdb.
      --output_top_path OUTPUT_TOP_PATH
                            Path to save the output topology file (pdb). Used if trajectory format requires separate topology. Accepted formats: pdb.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_results_npz_path** (*string*): Path to the input reconstructed results file (.npz), typically containing an 'xhat' array. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_results.npz). Accepted formats: NPZ
* **input_stats_pt_path** (*string*): Path to the input model statistics file (.pt) containing cartesian indices and optionally topology. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_input_model.pt). Accepted formats: PT
* **input_topology_path** (*string*): Path to the topology file (PDB) used if no suitable topology is found in the stats file. Used if no topology is found in stats. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/mdae/ref_input_topology.pdb). Accepted formats: PDB
* **output_traj_path** (*string*): Path to save the trajectory in xtc/pdb/dcd format. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.xtc). Accepted formats: XTC, PDB, DCD
* **output_top_path** (*string*): Path to save the output topology file (pdb). Used if trajectory format requires separate topology. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/mdae/output_model.pdb). Accepted formats: PDB
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **restart** (*boolean*): (False) Do not execute if output files exist.
### YAML
### JSON

## Lrp
Performs Layer-wise Relevance Propagation on a trained autoencoder encoder.
### Get help
Command:
```python
LRP -h
```
    usage: LRP [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH --input_dataset_pt_path INPUT_DATASET_PT_PATH [-o OUTPUT_RESULTS_NPZ_PATH]
    
    Performs Layer-wise Relevance Propagation on a trained autoencoder encoder.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the trained model file whose encoder is analyzed. Accepted formats: pth.
      --input_dataset_pt_path INPUT_DATASET_PT_PATH
                            Path to the input dataset file (.pt) used for computing relevance scores. Accepted formats: pt.
    
    optional arguments:
      -o OUTPUT_RESULTS_NPZ_PATH, --output_results_npz_path OUTPUT_RESULTS_NPZ_PATH
                            Path to the output results file containing relevance scores (compressed NumPy archive). Accepted formats: npz.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained model file whose encoder is analyzed. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) used for computing relevance scores. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt). Accepted formats: PT
* **output_results_npz_path** (*string*): Path to the output results file containing relevance scores (compressed NumPy archive). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_results.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Dataset** (*object*): ({}) Dataset/DataLoader options (e.g. batch_size and optional indices to subset the dataset).
### YAML
### JSON

## Make_plumed
Generate PLUMED input for biased dynamics using an MDAE model.
### Get help
Command:
```python
make_plumed -h
```
    usage: make_plumed [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH [--input_stats_pt_path INPUT_STATS_PT_PATH] [--input_reference_pdb_path INPUT_REFERENCE_PDB_PATH] [--input_ndx_path INPUT_NDX_PATH] --output_plumed_dat_path OUTPUT_PLUMED_DAT_PATH --output_features_dat_path OUTPUT_FEATURES_DAT_PATH --output_model_ptc_path OUTPUT_MODEL_PTC_PATH
    
    Generate PLUMED input for biased dynamics using an MDAE model.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the trained PyTorch model (.pth) to be converted to TorchScript and used in PLUMED. Accepted formats: pth.
      --output_plumed_dat_path OUTPUT_PLUMED_DAT_PATH
                            Path to the output PLUMED input file. Accepted formats: dat.
      --output_features_dat_path OUTPUT_FEATURES_DAT_PATH
                            Path to the output features.dat file describing the CVs to PLUMED. Accepted formats: dat.
      --output_model_ptc_path OUTPUT_MODEL_PTC_PATH
                            Path to the output TorchScript model file (.ptc) for PLUMED's PYTORCH_MODEL action. Accepted formats: ptc.
    
    optional arguments:
      --input_stats_pt_path INPUT_STATS_PT_PATH
                            Path to statistics file (.pt) produced during featurization, used to derive the PLUMED features.dat content. Accepted formats: pt.
      --input_reference_pdb_path INPUT_REFERENCE_PDB_PATH
                            Path to reference PDB used for FIT_TO_TEMPLATE actions when Cartesian features are present. Accepted formats: pdb.
      --input_ndx_path INPUT_NDX_PATH
                            Path to GROMACS index (NDX) file used to define groups when required by PLUMED. Accepted formats: ndx.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the trained PyTorch model (.pth) to be converted to TorchScript and used in PLUMED. File type: input. [Sample file](None). Accepted formats: PTH
* **input_stats_pt_path** (*string*): Path to statistics file (.pt) produced during featurization, used to derive the PLUMED features.dat content. File type: input. [Sample file](None). Accepted formats: PT
* **input_reference_pdb_path** (*string*): Path to reference PDB used for FIT_TO_TEMPLATE actions when Cartesian features are present. File type: input. [Sample file](None). Accepted formats: PDB
* **input_ndx_path** (*string*): Path to GROMACS index (NDX) file used to define groups when required by PLUMED. File type: input. [Sample file](None). Accepted formats: NDX
* **output_plumed_dat_path** (*string*): Path to the output PLUMED input file. File type: output. [Sample file](None). Accepted formats: DAT
* **output_features_dat_path** (*string*): Path to the output features.dat file describing the CVs to PLUMED. File type: output. [Sample file](None). Accepted formats: DAT
* **output_model_ptc_path** (*string*): Path to the output TorchScript model file (.ptc) for PLUMED's PYTORCH_MODEL action. File type: output. [Sample file](None). Accepted formats: PTC
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **include_energy** (*boolean*): (True) Whether to include ENERGY in PLUMED.
* **bias** (*array*): ([]) List of biasing actions (e.g. METAD) to be added to the PLUMED file.
* **prints** (*object*): ({'ARG': '*', 'STRIDE': 1, 'FILE': 'COLVAR'}) PRINT command parameters (e.g. ARG, STRIDE, FILE).
* **group** (*object*): ({}) GROUP definition options (label, NDX group or atom selection parameters).
* **wholemolecules** (*object*): ({}) WHOLEMOLECULES options when using Cartesian coordinates.
* **fit_to_template** (*object*): ({}) FIT_TO_TEMPLATE options (e.g. STRIDE, TYPE, etc.).
* **pytorch_model** (*object*): ({}) PYTORCH_MODEL options (label, PACE and other parameters).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_make_plumed.yml)
```python
properties:
  additional_actions:
  - label: ene
    name: ENERGY
  bias:
  - label: bias
    name: METAD
    params:
      ARG: cv.1
      BIASFACTOR: 8
      FILE: HILLS
      HEIGHT: 1.2
      PACE: 500
      SIGMA: 0.35
  fit_to_template:
    STRIDE: 1
    TYPE: OPTIMAL
  group:
    NDX_GROUP: chA_&_C-alpha
    label: c_alphas
  prints:
    ARG: cv.*,bias.*
    FILE: COLVAR
    STRIDE: 1
  pytorch_model:
    PACE: 1
    label: cv
  wholemolecules:
    ENTITY0: c_alphas

```
#### Command line
```python
make_plumed --config config_make_plumed.yml --input_model_pth_path input.pth --input_stats_pt_path input.pt --input_reference_pdb_path input.pdb --input_ndx_path input.ndx --output_plumed_dat_path output.dat --output_features_dat_path output.dat --output_model_ptc_path output.ptc
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_make_plumed.json)
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
make_plumed --config config_make_plumed.json --input_model_pth_path input.pth --input_stats_pt_path input.pt --input_reference_pdb_path input.pdb --input_ndx_path input.ndx --output_plumed_dat_path output.dat --output_features_dat_path output.dat --output_model_ptc_path output.ptc
```

## Mdfeaturizer
Obtain the Molecular Dynamics Features for PyTorch model training.
### Get help
Command:
```python
MDFeaturizer -h
```
    usage: MDFeaturizer [-h] [-c CONFIG] [--input_trajectory_path INPUT_TRAJECTORY_PATH] --input_topology_path INPUT_TOPOLOGY_PATH --output_dataset_pt_path OUTPUT_DATASET_PT_PATH --output_stats_pt_path OUTPUT_STATS_PT_PATH
    
    Obtain the Molecular Dynamics Features for PyTorch model training.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_topology_path INPUT_TOPOLOGY_PATH
                            Path to the input topology file. Accepted formats: pdb.
      --output_dataset_pt_path OUTPUT_DATASET_PT_PATH
                            Path to the output dataset model file. Accepted formats: pt.
      --output_stats_pt_path OUTPUT_STATS_PT_PATH
                            Path to the output model statistics file. Accepted formats: pt.
    
    optional arguments:
      --input_trajectory_path INPUT_TRAJECTORY_PATH
                            Path to the input trajectory file (if omitted topology file is used as trajectory). Accepted formats: xtc, dcd.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_trajectory_path** (*string*): Path to the input trajectory file (if omitted topology file is used as trajectory). File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.xtc). Accepted formats: XTC, DCD
* **input_topology_path** (*string*): Path to the input topology file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/MCV1900209.pdb). Accepted formats: PDB
* **output_dataset_pt_path** (*string*): Path to the output dataset model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_dataset.pt). Accepted formats: PT
* **output_stats_pt_path** (*string*): Path to the output model statistics file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_stats.pt). Accepted formats: PT
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **cartesian** (*object*): ({'selection': 'name CA'}) Atom selection options for Cartesian coordinates feature generation (e.g. selection, fit_selection).
* **distances** (*object*): ({'selection': 'name CA', 'cutoff': 0.4, 'periodic': True, 'bonded': False}) Atom selection options for pairwise distance features (selection, cutoff, periodic, bonded, etc.).
* **angles** (*object*): ({'selection': 'backbone', 'periodic': True, 'bonded': True}) Atom selection options for angle features (selection, periodic, bonded, etc.).
* **dihedrals** (*object*): ({'selection': 'backbone', 'periodic': True, 'bonded': True}) Atom selection options for dihedral features (selection, periodic, bonded, etc.).
* **options** (*object*): ({'norm_in': 'min_max'}) General processing options (e.g. timelag, norm_in).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_mdfeaturizer.yml)
```python
properties:
  cartesian:
    selection: name CA
  distances:
    cutoff: 0.4
    periodic: true
    selection: name CA
  options:
    norm_in:
      mode: min_max

```
#### Command line
```python
MDFeaturizer --config config_mdfeaturizer.yml --input_trajectory_path train_mdae_traj.xtc --input_topology_path MCV1900209.pdb --output_dataset_pt_path ref_output_dataset.pt --output_stats_pt_path ref_output_stats.pt
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_mdfeaturizer.json)
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
      "norm_in": {
        "mode": "min_max"
      }
    }
  }
}
```
#### Command line
```python
MDFeaturizer --config config_mdfeaturizer.json --input_trajectory_path train_mdae_traj.xtc --input_topology_path MCV1900209.pdb --output_dataset_pt_path ref_output_dataset.pt --output_stats_pt_path ref_output_stats.pt
```

## trainModel
Trains a PyTorch autoencoder using the given properties.
### Get help
Command:
```python
trainModel -h
```
    usage: trainModel [-h] [-c CONFIG] --input_model_pth_path INPUT_MODEL_PTH_PATH --input_dataset_pt_path INPUT_DATASET_PT_PATH [--output_model_pth_path OUTPUT_MODEL_PTH_PATH] [--output_metrics_npz_path OUTPUT_METRICS_NPZ_PATH]
    
    Trains a PyTorch autoencoder using the given properties.
    
    options:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            This file can be a YAML file, JSON file or JSON string
    
    required arguments:
      --input_model_pth_path INPUT_MODEL_PTH_PATH
                            Path to the input model file. Accepted formats: pth.
      --input_dataset_pt_path INPUT_DATASET_PT_PATH
                            Path to the input dataset file (.pt) produced by the MD feature pipeline. Accepted formats: pt.
    
    optional arguments:
      --output_model_pth_path OUTPUT_MODEL_PTH_PATH
                            Path to save the trained model (.pth). If omitted, the trained model is only available in memory. Accepted formats: pth.
      --output_metrics_npz_path OUTPUT_METRICS_NPZ_PATH
                            Path save training metrics in compressed NumPy format (.npz). Accepted formats: npz.
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_model_pth_path** (*string*): Path to the input model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **input_dataset_pt_path** (*string*): Path to the input dataset file (.pt) produced by the MD feature pipeline. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pt). Accepted formats: PT
* **output_model_pth_path** (*string*): Path to save the trained model (.pth). If omitted, the trained model is only available in memory. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.pth). Accepted formats: PTH
* **output_metrics_npz_path** (*string*): Path save training metrics in compressed NumPy format (.npz). File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/output_model.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **Trainer** (*object*): ({}) PyTorch Lightning Trainer options (e.g. max_epochs, callbacks, logger, profiler, accelerator, devices, etc.).
* **Dataset** (*object*): ({}) mlcolvar DictDataset / DictModule options (e.g. batch_size, split proportions and shuffling flags).
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_model.yml)
```python
properties:
  Dataset:
    batch_size: 32
    split:
      train_prop: 0.8
      val_prop: 0.2
  Trainer:
    callbacks:
      metrics:
      - EarlyStopping
    max_epochs: 10

```
#### Command line
```python
trainModel --config config_train_model.yml --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_model_pth_path output_model.pth --output_metrics_npz_path output_model.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_model.json)
```python
{
  "properties": {
    "Trainer": {
      "max_epochs": 10,
      "callbacks": {
        "metrics": [
          "EarlyStopping"
        ]
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
train_model --config config_train_model.json --input_model_pth_path output_model.pth --input_dataset_pt_path output_model.pt --output_model_pth_path output_model.pth --output_metrics_npz_path output_model.npz
```
