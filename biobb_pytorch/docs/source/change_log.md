# Biobb PYTORCH Change Log

All notable changes to this project will be documented in this file.

## [Unreleased]

## What's new in version [5.4.0](https://github.com/bioexcel/biobb_pytorch/releases/tag/v5.4.0)?

### Changes

* [UPDATE] Update to biobb_common 5.4.0.
* [FIX] mdfeaturizer and make_plumed featurize the trajectory and load the statistics when the block runs, after the input and restart checks, not when it is created: with `restart`, existing outputs are no longer featurized again.
* [FIX] LRP works on machines with a GPU: the data were moved to the GPU but the model stayed on the CPU, and the block failed.
* [FIX] `build_model`: `model_type`, `n_cvs`, `encoder_layers`, `decoder_layers` and `options` take their documented defaults when they are not given (they were required: "Missing required properties").
* [FIX] `train_model`, `evaluate_model`, `encode_model` and `decode_model`: the `Trainer` and `Dataset` properties, and the `split` of `Dataset`, can be left out, as documented (a `KeyError` before).
* [FIX] `feat2traj`: `output_top_path` is optional, as documented: an XTC or DCD trajectory without it no longer fails. A statistics file without cartesian indices stops with "No cartesian indices found in stats file.".
* [FIX] `mdfeaturizer`: the statistics are saved to the given `output_stats_pt_path` (its extension was replaced by `.pt`).
* [FIX] `make_plumed`: without `input_stats_pt_path` it no longer fails with an `AttributeError` when it is created.
* [UPDATE] Python 3.11 to 3.13.

## What's new in version [5.3.0](https://github.com/bioexcel/biobb_amber/releases/tag/v5.3.0)?

### Changes

* [UPDATE] Update to biobb_common 5.3.1

## What's new in version [5.2.3](https://github.com/bioexcel/biobb_amber/releases/tag/v5.2.3)?

### Changes

* [FEATURE] Bug fixes in executable names.

## What's new in version [5.2.2](https://github.com/bioexcel/biobb_amber/releases/tag/v5.2.2)?

### Changes

* [UPDATE] Update to biobb_common 5.2.2

## What's new in version [5.2.1](https://github.com/bioexcel/biobb_amber/releases/tag/v5.2.1)?

### Changes

* [UPDATE] Update to torch 2.5.0
* [FEATURE] Minor bug fixes in make_plumed.

## What's new in version [5.2.0](https://github.com/bioexcel/biobb_amber/releases/tag/v5.2.0)?

### Changes

* [UPDATE] Update to biobb_common 5.2.0
* [FEATURE] Add new blocks: build_mdae, decode_model, enconde_model, train_mdae, feat2traj, LRP, make_plumed and mdfeaturizer.

## What's new in version [5.1.0](https://github.com/bioexcel/biobb_pytorch/releases/tag/v5.1.0)?

### Changes

* [UPDATE]: Update biobb_common to 5.1.0

## What's new in version [5.0.0](https://github.com/bioexcel/biobb_pytorch/releases/tag/v5.0.0)?

### Changes

* [CI/CD](linting_and_testing.yml): Update set-up micromamba.
* [CI/CD](conf.yml): Adding global properties to test yaml configuration
* [CI/CD](linting_and_testing.yaml): Update GA test workflow to Python >3.9
* [DOCS](.readthedocs.yaml): Updating to Python 3.9
* [CI/CD](GITIGNORE): Update .gitignore to include the new file extensions to ignore
* [CI/CD](conf.yml): Change test conf.yml to adapt to new settings configuration

## What's new in version [4.2.1](https://github.com/bioexcel/biobb_pytorch/releases/tag/v5.0.0)?

### Changes

* [CI/CD] Updating checkout GA action to v4
* [TESTS] Avoiding type hint checking in test files
* [CI/CD] Add sync workflow for CASTIEL project
* [DOCS] Updating command line documentation
* [FIX] Adding shebang to the file first line
* [FIX] Adding mypy_cache to gitignore
* [DOCS] Update Bioschemas

## What's new in version [4.2.0](https://github.com/bioexcel/biobb_pytorch/releases/tag/v4.2.0)?

### Changes

* [DOCS] Adding fair software badge and GA
* [TESTS] Disabling equality tests for CI/CD pipelines
* [TEST] Update both tests and test data to use a common seed
* [FIX] Adding .vscode to gitignore
* [FIX] Taking apart the training and evaluation functions to avoid artifacs
* [FIX] Converting nm to Angstroms in plots
* [FEATURE] Adding Change log file

## What's new in version [4.1.4](https://github.com/bioexcel/biobb_pytorch/releases/tag/v4.1.3)?

### New features:

- Added new seed parameter to the `biobb_pytorch.pytorch.train` module to set the seed for the random number generator. Default value is None.

