# Revisiting Deep Learning for Variable Type Recovery

[**Paper PDF**](https://arxiv.org/pdf/2304.03854.pdf)

This repository hosts the code used to replicate the paper [Augmenting Decompiler Output with Learned Variable Names and Types](https://cmustrudel.github.io/papers/ChenDIRTY2022.pdf).
It is a fork of the [original DIRTY implementation](https://github.com/CMUSTRUDEL/DIRTY) written by Chen et. al.
While most of the model code remains identical, we add support for generating a training dataset with the [Ghidra decompiler](https://github.com/NationalSecurityAgency/ghidra), allowing researchers without an IDA Pro license to train their own DIRTY model.
The original README provides clear instructions on how to download and run their pre-trained DIRTY model, but the README's instructions are slightly unclear when describing how to train your own model.
This README explicitly covers all the steps necessary to train a DIRTY model from scratch.

## Getting Started with DIRTY-Ghidra Inference

Most people probably just want to use DIRTY-Ghidra to predict variable names and
types for their own binaries.  If that is you, follow these instructions:

1. Clone this repository to `DIRTY_DIR`
2. Create a virtual environment (venv) and install the requirements via `pip install -r requirements.txt`
3. Install Ghidra
4. Install Ghidrathon.  Make sure you configure Ghidrathon (`python
   ghidrathon_configure.py`) using the venv from step 2.
5. Download [data1.tar.bz2](https://cmu.box.com/s/nx9fyn8jx0i9p4bftw8f2giqlufnoyj5) and extract it in DIRTY_DIR/dirty (`tar -xvjf data1.tar.bz2 -C DIRTY_DIR/dirty`)
6. Run `mkdir ~/ghidra_scripts && ln -s DIRTY_DIR/scripts/DIRTY_infer.py ~/ghidra_scripts/DIRTY_infer.py` if on Linux.
7. Open a function in Ghidra.  Run the script `DIRTY_infer.py` in the script manager.
8. Optionally assign the script to a keyboard shortcut.

## Requirements

- Linux with Python 3.10+
- [PyTorch ≥ 1.5.1](https://pytorch.org/)
- [Ghidrathon >= 4.0.0](https://github.com/mandiant/Ghidrathon)
- `pip install -r requirements.txt`

### Libraries

A few libraries are required by the python packages.  On ubuntu, you can install
these with:
- `apt install pkg-config libsentencepiece-dev`

## Training a DIRTY model

### Dataset Generation
The first step to train DIRTY is to obtain a unprocessed DIRT dataset. Instructions can be found in the `dataset-gen-ghidra` folder.

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
