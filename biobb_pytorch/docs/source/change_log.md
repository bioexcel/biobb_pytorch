# Biobb PYTORCH Change Log

All notable changes to this project will be documented in this file.

## What's new in version [4.2.1](https://github.com/bioexcel/biobb_pytorch/releases/tag/v4.2.1)?

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

