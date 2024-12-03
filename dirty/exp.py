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
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    BatchSizeFinder,
    LearningRateMonitor,
)
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader

from model.model import TypeReconstructionModel
from utils import util
from utils.dataset import Dataset


def train(args):
    config = json.loads(_jsonnet.evaluate_file(args["CONFIG_FILE"]))

    if args["--extra-config"]:
        extra_config = args["--extra-config"]
        extra_config = json.loads(extra_config)
        config = util.update(config, extra_config)

    # dataloaders
    batch_size = (
        config["test"]["batch_size"]
        if args["--eval-ckpt"]
        else config["train"]["batch_size"]
    )
    train_set = Dataset(
        config["data"]["train_file"], config["data"], percent=float(args["--percent"])
    )
    test_set = Dataset(config["data"]["test_file"], config["data"])
    dev_set = Dataset(config["data"]["dev_file"], config["data"])

    print(f"Length of training dataset is {len(train_set)} examples")

    # Define DataModule for batch finding.
    class LitDataModule(LightningDataModule):
        def __init__(self, batch_size=batch_size):
            super().__init__()
            self.batch_size = batch_size

        def test_dataloader(self):
            return DataLoader(
                test_set,
                batch_size=self.batch_size,
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

    if "torch_float32_matmul" in config["train"]:
        torch.set_float32_matmul_precision(config["train"]["torch_float32_matmul"])

    wandb_logger = WandbLogger(name=args["--expname"], project="dire", log_model="all")
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(model, log="all", log_freq=10000)
    monitor_var = "val_acc"
    resume_from_checkpoint = (
        args["--eval-ckpt"] if args["--eval-ckpt"] else args["--resume"]
    )
    if resume_from_checkpoint == "":
        resume_from_checkpoint = None

    # Adds a safety margin.  For example, `safety_margin`  of 0.1 indicates that
    # the final batch_size will be reduced by 10%
    class SafeBatchSizeFinder(BatchSizeFinder):
        def __init__(self, safety_margin=0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert safety_margin >= 0 and safety_margin <= 1.0
            self.safety_margin = safety_margin

        def scale_batch_size(self, trainer, *args, **kwargs):
            super().scale_batch_size(trainer, *args, **kwargs)
            original_batch_size = self.optimal_batch_size
            new_batch_size = int(self.optimal_batch_size * (1.0 - self.safety_margin))
            print(
                f"Found optimal batch size of {original_batch_size}, but with a safety margin of {self.safety_margin}, reducing it to {new_batch_size}"
            )
            self.optimal_batch_size = new_batch_size
            # This adjusts the data module batch_size.
            pl.tuner.batch_size_scaling._adjust_batch_size(
                trainer, value=new_batch_size
            )
            pl.tuner.batch_size_scaling._reset_dataloaders(trainer)
            trainer._active_loop.reset()

    trainer = pl.Trainer(
        precision=config["train"].get("precision", 32),
        max_epochs=config["train"]["max_epoch"],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        callbacks=[
            EarlyStopping(
                monitor=monitor_var,
                mode="max",
                patience=config["train"]["patience"],
            ),
            # Save all checkpoints that improve accuracy
            ModelCheckpoint(
                monitor=monitor_var,
                filename="{epoch}-{%s:.2f}" % monitor_var,
                save_top_k=2,
                mode="max",
            ),
            SafeBatchSizeFinder(
                safety_margin=config["train"].get("safety_margin", 0.1),
                init_val=batch_size,
                max_trials=30,
                steps_per_trial=3,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        check_val_every_n_epoch=config["train"]["check_val_every_n_epoch"],
        accumulate_grad_batches=config["train"]["grad_accum_step"],
        limit_test_batches=(
            config["test"]["limit"] if "limit" in config["test"] else 1.0
        ),
    )

    datamodule = LitDataModule(batch_size=batch_size)

    if args["--eval-ckpt"]:
        ret = trainer.test(model, datamodule=datamodule, ckpt_path=args["--eval-ckpt"])
        json.dump(ret[0], open("test_result.json", "w"))
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint)


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
