# BioBB PYTORCH Command Line Help
Generic usage:
```python
biobb_command [-h] --config CONFIG --input_file(s) <input_file(s)> --output_file <output_file>
```
-----------------


## Train_mdae
Train a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
train_mdae -h
```
    /bin/sh: train_mdae: command not found
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_train_npy_path** (*string*): Path to the input train data file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.npy). Accepted formats: NPY
* **output_model_pth_path** (*string*): Path to the output model file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth). Accepted formats: PTH
* **input_model_pth_path** (*string*): Path to the input model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth). Accepted formats: PTH
* **output_train_data_npz_path** (*string*): Path to the output train data file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_train_data.npz). Accepted formats: NPZ
* **output_performance_npz_path** (*string*): Path to the output performance file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_performance.npz). Accepted formats: NPZ
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **latent_dimensions** (*integer*): (2) min dimensionality of the latent space..
* **num_layers** (*integer*): (4) number of layers in the encoder/decoder (4 to encode and 4 to decode)..
* **num_epochs** (*integer*): (100) number of epochs (iterations of whole dataset) for training..
* **lr** (*number*): (0.0001) learning rate..
* **lr_step_size** (*integer*): (100) Period of learning rate decay..
* **gamma** (*number*): (0.1) Multiplicative factor of learning rate decay..
* **checkpoint_interval** (*integer*): (25) number of epochs interval to save model checkpoints o 0 to disable..
* **output_checkpoint_prefix** (*string*): (checkpoint_epoch) prefix for the checkpoint files..
* **partition** (*number*): (0.8) 0.8 = 80% partition of the data for training and validation..
* **batch_size** (*integer*): (1) number of samples/frames per batch..
* **log_interval** (*integer*): (10) number of epochs interval to log the training progress..
* **input_dimensions** (*integer*): (None) input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)..
* **output_dimensions** (*integer*): (None) output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)..
* **loss_function** (*string*): (MSELoss) Loss function to be used. .
* **optimizer** (*string*): (Adam) Optimizer algorithm to be used. .
* **seed** (*integer*): (None) Random seed for reproducibility..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_mdae.yml)
```python
properties:
  num_epochs: 50
  seed: 1

```
#### Command line
```python
train_mdae --config config_train_mdae.yml --input_train_npy_path train_mdae_traj.npy --output_model_pth_path ref_output_model.pth --input_model_pth_path ref_output_model.pth --output_train_data_npz_path ref_output_train_data.npz --output_performance_npz_path ref_output_performance.npz
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_train_mdae.json)
```python
{
  "properties": {
    "num_epochs": 50,
    "seed": 1
  }
}
```
#### Command line
```python
train_mdae --config config_train_mdae.json --input_train_npy_path train_mdae_traj.npy --output_model_pth_path ref_output_model.pth --input_model_pth_path ref_output_model.pth --output_train_data_npz_path ref_output_train_data.npz --output_performance_npz_path ref_output_performance.npz
```

## Apply_mdae
Apply a Molecular Dynamics AutoEncoder (MDAE) PyTorch model.
### Get help
Command:
```python
apply_mdae -h
```
    /bin/sh: apply_mdae: command not found
### I / O Arguments
Syntax: input_argument (datatype) : Definition

Config input / output arguments for this building block:
* **input_data_npy_path** (*string*): Path to the input data file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/data/mdae/train_mdae_traj.npy). Accepted formats: NPY
* **input_model_pth_path** (*string*): Path to the input model file. File type: input. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_model.pth). Accepted formats: PTH
* **output_reconstructed_data_npy_path** (*string*): Path to the output reconstructed data file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_reconstructed_data.npy). Accepted formats: NPY
* **output_latent_space_npy_path** (*string*): Path to the reduced dimensionality file. File type: output. [Sample file](https://github.com/bioexcel/biobb_pytorch/raw/master/biobb_pytorch/test/reference/mdae/ref_output_latent_space.npy). Accepted formats: NPY
### Config
Syntax: input_parameter (datatype) - (default_value) Definition

Config parameters for this building block:
* **batch_size** (*integer*): (1) number of samples/frames per batch..
* **latent_dimensions** (*integer*): (2) min dimensionality of the latent space..
* **num_layers** (*integer*): (4) number of layers in the encoder/decoder (4 to encode and 4 to decode)..
* **input_dimensions** (*integer*): (None) input dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)..
* **output_dimensions** (*integer*): (None) output dimensions by default it should be the number of features in the input data (number of atoms * 3 corresponding to x, y, z coordinates)..
### YAML
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_apply_mdae.yml)
```python
properties:
  batch_size: 1

```
#### Command line
```python
apply_mdae --config config_apply_mdae.yml --input_data_npy_path train_mdae_traj.npy --input_model_pth_path ref_output_model.pth --output_reconstructed_data_npy_path ref_output_reconstructed_data.npy --output_latent_space_npy_path ref_output_latent_space.npy
```
### JSON
#### [Common config file](https://github.com/bioexcel/biobb_pytorch/blob/master/biobb_pytorch/test/data/config/config_apply_mdae.json)
```python
{
  "properties": {
    "batch_size": 1
  }
}
```
#### Command line
```python
apply_mdae --config config_apply_mdae.json --input_data_npy_path train_mdae_traj.npy --input_model_pth_path ref_output_model.pth --output_reconstructed_data_npy_path ref_output_reconstructed_data.npy --output_latent_space_npy_path ref_output_latent_space.npy
```
