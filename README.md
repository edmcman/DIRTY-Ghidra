# Revisiting Deep Learning for Variable Type Recovery

[**Paper PDF**](https://arxiv.org/pdf/2304.03854.pdf)

This repository hosts the code used to replicate the paper [Augmenting Decompiler Output with Learned Variable Names and Types](https://cmustrudel.github.io/papers/ChenDIRTY2022.pdf).
It is a fork of the [original DIRTY implementation](https://github.com/CMUSTRUDEL/DIRTY) written by Chen et. al.
While most of the model code remains identical, we add support for generating a training dataset with the [Ghidra decompiler](https://github.com/NationalSecurityAgency/ghidra), allowing researchers without an IDA Pro license to train their own DIRTY model.
The original README provides clear instructions on how to download and run their pre-trained DIRTY model, but the README's instructions are slightly unclear when describing how to train your own model.
This README explicitly covers all the steps necessary to train a DIRTY model from scratch.

This is @edmcman's fork of the original DIRTY-Ghidra repository.  It features a number of improvements and bug fixes, and also includes the ability to perform inference on new examples.

## Getting Started with DIRTY-Ghidra Inference

[![Test DIRTY Ghidra's inference ability](https://github.com/edmcman/DIRTY-Ghidra/actions/workflows/test.yml/badge.svg)](https://github.com/edmcman/DIRTY-Ghidra/actions/workflows/test.yml)

Most people probably just want to use DIRTY-Ghidra to predict variable names and
types for their own binaries.  If that is you, follow these instructions:

1. Clone this repository to `DIRTY_DIR`
2. Optional but highly recommended: Create a virtual environment (venv) with `python -m venv /path/to/venv; source /path/to/venv/bin/activate`. This will prevent DIRTY from interfering with your system python packages.
3. Install the requirements via `pip install -r requirements.txt`
4. [Install Ghidra](https://ghidra-sre.org/InstallationGuide.html)
5. [Install Ghidrathon](https://github.com/mandiant/Ghidrathon/?tab=readme-ov-file#installing-ghidrathon).  Make sure you configure Ghidrathon (`python
   ghidrathon_configure.py`) using the venv from step 2.
6. Download the latest model from HF (`huggingface_hub[cli] && huggingface-cli download --repo-type model ejschwartz/dirty-ghidra --local-dir $DIRTY_DIR/dirty`)
7. Run `mkdir ~/ghidra_scripts && ln -s DIRTY_DIR/scripts/DIRTY_infer.py ~/ghidra_scripts/DIRTY_infer.py` if on Linux.
8. Open a function in Ghidra.  Run the script `DIRTY_infer.py` in the script manager.
9. Optionally assign the script to a keyboard shortcut.

## Requirements

- Linux with Python 3.10+
- [PyTorch ≥ 1.5.1](https://pytorch.org/)
- [Ghidrathon >= 4.0.0](https://github.com/mandiant/Ghidrathon)
- `pip install -r requirements.txt`

### Libraries

A few libraries are required by the python packages.  On ubuntu, you can install
these with:
- `apt install pkg-config libsentencepiece-dev libprotobuf-dev`

## Training a DIRTY model

### Dataset Generation
The first step to train DIRTY is to obtain a unprocessed DIRT dataset. Instructions can be found in the [dataset-gen-ghidra](dataset-gen-ghidra) folder.

### Dataset Preprocessing

Once we have a unprocessed dataset, we want to preprocess the dataset to generate the training samples the model will train on.

```bash
# inside the `dirty` directory
python3 -m utils.preprocess [-h] [options] INPUT_FOLDER INPUT_FNAMES TARGET_FOLDER
```

Given the path to the `INPUT_FOLDER` that contains the unprocessed dataset and the path to the `INPUT_FNAMES` that contains the names of all files you want to process, this script creates the preprocessed dataset in `TARGET_FOLDER`.
`TARGET_FOLDER` will contain the following files:
- train-shard-\*.tar : archive of the training dataset samples
- dev.tar : archive of the validation dataset
- test.tar : archive of the test dataset 
- typelib.json: list of all types contained across the three datasets

We also need to build a vocabulary of tokens that the model will understand

```bash
# inside the `dirty` directory
python3 -m utils.vocab [-h] --use-bpe [options] TRAIN_FILES_TAR PATH_TO_TYPELIB_JSON TARGET_DIRECTORY/vocab.bpe10000
```

This script generates vocabulary files located in `TARGET_DIRECTORY`. It is recommended to prefix the vocab files with `vocab.bpe10000` to match the expected vocabulary filenames in the model config files.

Finally, lets move our dataset and vocabulary files to the directory expected by the model config files.

```bash
# inside the `dirty` directory
mkdir -p data1/
mv PATH_TO_TRAIN_SHARDS_TAR PATH_TO_DEV_TAR PATH_TO_TEST_TAR PATH_TO_VOCAB_BPE10000 data1/
```

We can now train our own DIRTY model and test its performance. Follow the steps starting at the [Train DIRTY section of the original README](https://github.com/CMUSTRUDEL/DIRTY/blob/main/README.md#train-dirty)
