"""
Experiment strip

Usage:
    exp.py train [options] CONFIG_FILE
    exp.py test [options] MODEL_FILE TEST_DATA_FILE

Options:
    -h --help                                   Show this screen
    --debug                                     Debug mode
    --seed=<int>                                Seed [default: 0]
    --expname=<str>                             work dir [default: type]
    --eval-ckpt=<str>                           load checkpoint for eval [default: ]
    --resume=<str>                              load checkpoint for resume training [default: ]
    --extra-config=<str>                        extra config [default: {}]
    --percent=<float>                           percent of training data used [default: 1.0]
"""
import json
import os
import random
import sys
from typing import Dict, Iterable, List, Tuple

import _jsonnet
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from docopt import docopt
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader

from model.model import TypeReconstructionModel
from utils import util
from utils.dataset import Dataset, WrappedLenDataset


def train(args):
    config = json.loads(_jsonnet.evaluate_file(args["CONFIG_FILE"]))

    if args["--extra-config"]:
        extra_config = args["--extra-config"]
        extra_config = json.loads(extra_config)
        config = util.update(config, extra_config)

    # dataloaders
    batch_size = config["train"]["batch_size"]
    train_set = Dataset(
        config["data"]["train_file"], config["data"], percent=float(args["--percent"])
    )
    train_set = WrappedLenDataset(train_set)
    test_set = Dataset(config["data"]["test_file"], config["data"])
    test_set = WrappedLenDataset(test_set)
    dev_set = Dataset(config["data"]["dev_file"], config["data"])

    print(f"len is {len(train_set)}")

    # Define DataModule for batch finding.
    class LitDataModule(LightningDataModule):
        def __init__(self, batch_size = batch_size):
            super().__init__()
            self.batch_size = batch_size

        def test_dataloader(self):
            return DataLoader(
                test_set,
                batch_size=config["test"]["batch_size"],
                collate_fn=Dataset.collate_fn,
                num_workers=8,
                pin_memory=True,
            )

        def train_dataloader(self):
            return DataLoader(
                train_set,
                batch_size=self.batch_size,
                collate_fn=Dataset.collate_fn,
                num_workers=16,
                pin_memory=True,
            )

        def val_dataloader(self):
            return DataLoader(
                dev_set,
                batch_size=self.batch_size,
                collate_fn=Dataset.collate_fn,
                num_workers=8,
                pin_memory=True,
            )

    # model
    model = TypeReconstructionModel(config)

    wandb_logger = WandbLogger(name=args["--expname"], project="dire", log_model=True)
    wandb_logger.log_hyperparams(config)
    resume_from_checkpoint = (
        args["--eval-ckpt"] if args["--eval-ckpt"] else args["--resume"]
    )
    if resume_from_checkpoint == "":
        resume_from_checkpoint = None
    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epoch"],
        logger=wandb_logger,
        gradient_clip_val=1,
        callbacks=[
            EarlyStopping(
                monitor="val_retype_acc"
                if config["data"]["retype"]
                else "val_rename_acc",
                mode="max",
                patience=config["train"]["patience"],
            )
        ],
        check_val_every_n_epoch=config["train"]["check_val_every_n_epoch"],
        accumulate_grad_batches=config["train"]["grad_accum_step"],
        limit_test_batches=config["test"]["limit"] if "limit" in config["test"] else 1.0
    )

    datamodule = LitDataModule(batch_size=batch_size)

    if args["--eval-ckpt"]:
        # HACK: necessary to make pl test work for IterableDataset
        Dataset.__len__ = lambda self: 1000000
        ret = trainer.test(model, test_dataloaders=datamodule.test_dataloader(), ckpt_path=args["--eval-ckpt"])
        json.dump(ret[0], open("test_result.json", "w"))
    else:
        try:
            tuner = Tuner(trainer)
            tuner.scale_batch_size(model, init_val=batch_size, datamodule=datamodule, max_trials=10)
            print(f"Largest batch size: {datamodule.batch_size}")
        except pl.utilities.exceptions.MisconfigurationException:
            print("Couldn't find largest batch size")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    cmd_args = docopt(__doc__)
    print(f"Main process id {os.getpid()}", file=sys.stderr)

    # seed the RNG
    seed = int(cmd_args["--seed"])
    print(f"use random seed {seed}", file=sys.stderr)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    random.seed(seed * 17 // 7)

    if cmd_args["train"]:
        train(cmd_args)
