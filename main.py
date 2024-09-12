import argparse
import datetime
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch

from dataset.batch_gen import BatchGenerator, TestSampleGenerator
from dataset.diffact_data import get_data_loader
from config import cfg, update_config
from model.asformer import ASFormerTrainer
from model.mstcn import MSTCNTrainer
from model.diffact import DiffActTrainer
from model.c2f import C2FTrainer
from utils.misc import read_actions, seconds_to_hours, set_seed


def get_train_dataset_loader(cfg):
    if cfg.MODEL.NAME == "diffact":
        return get_data_loader(cfg, mode="train")
    else:
        batch_gen = BatchGenerator(cfg)
        batch_gen.read_data(cfg.DATA.VID_LIST_FILE)
        return batch_gen
    

def get_test_dataset_loader(cfg):
    if cfg.MODEL.NAME == "diffact":
        return get_data_loader(cfg, mode="test")
    else:
        test_gen = TestSampleGenerator(cfg)
        test_gen.read_data(cfg.DATA.VID_LIST_FILE_TEST)
        return test_gen
    

def get_trainer(cfg):
    if cfg.MODEL.NAME == "mstcn":
        return MSTCNTrainer(cfg)
    elif cfg.MODEL.NAME == "asformer":
        return ASFormerTrainer(cfg)
    elif cfg.MODEL.NAME == "diffact":
        return DiffActTrainer(cfg)
    elif cfg.MODEL.NAME == "c2f":
        return C2FTrainer(cfg)


def train(cfg):
    set_seed(cfg.TRAIN.SEED)
    trainer = get_trainer(cfg)

    logging.info(f"------ Training ---------")
    train_dataset_loader = get_train_dataset_loader(cfg)
    start_time = time.time()
    logging.info(f"Starting time: {start_time}")
    trainer.train(train_dataset_loader)

    logging.info("------ Evaluation ---------")
    test_dataset_loader = get_test_dataset_loader(cfg)
    scores = trainer.predict(test_dataset_loader)
    logging.info(f"{scores}")

    end_time = time.time()
    logging.info(f"Ending time: {end_time}")

    cost_time = end_time - start_time
    hrs, mins, secs = seconds_to_hours(cost_time)
    logging.info(f"Total time: {hrs:.0f}h:{mins:.0f}m:{secs:.2f}s")


def eval(cfg):
    trainer = get_trainer(cfg)
    test_dataset_loader = get_test_dataset_loader(cfg)

    if cfg.TRAIN.EVAL_CHCKPTS:
         # eval all checkpointed epochs
        chckpt_epochs = list(range(cfg.TRAIN.CHCKPT_FREQ, cfg.TRAIN.NUM_EPOCHS, cfg.TRAIN.CHCKPT_FREQ))
    else:
        chckpt_epochs = []
    # always eval the last epoch's model
    chckpt_epochs.append(cfg.TRAIN.NUM_EPOCHS)

    logging.info("------ Evaluation ---------")
    res_dir, res_file = os.path.split(cfg.TRAIN.RES_FILENAME)
    for chckpt_epoch in chckpt_epochs:
        logging.info(f"Loading model saved at epoch {chckpt_epoch}")
        trainer.model.load_state_dict(
            torch.load(f"{cfg.TRAIN.MODEL_DIR}/epoch-{chckpt_epoch}.model"))
        scores = trainer.predict(test_dataset_loader, 
                                 os.path.join(res_dir, f"epoch_{chckpt_epoch}_" + res_file))
        logging.info(f"{scores}")


def extra_train_config(cfg):
    cfg.DATA.VID_LIST_FILE = (
        f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/splits/train.split{cfg.DATA.SPLIT}.bundle"
    )
    cfg.DATA.VID_LIST_FILE_TEST = (
        f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/splits/test.split{cfg.DATA.SPLIT}.bundle"
    )
    cfg.DATA.FEATURES_PATH = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/features/"
    cfg.DATA.GT_PATH = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/groundTruth/"
    cfg.DATA.MAPPING_FILE = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/mapping.txt"
    cfg.DATA.ACTIONS_DICT, cfg.DATA.NUM_CLASSES = read_actions(cfg.DATA.MAPPING_FILE)

    # exp string to indicate the setups, which could be 'standard', 'semi-supervised', etc.
    # cfg.TRAIN.SETUP = f""

    time_string = datetime.datetime.now().strftime("%m%d-%H-%M-%S-%f")[:-3]
    cfg.TRAIN.EXP_NAME = (
        f"{cfg.DATA.DATASET}_{cfg.MODEL.NAME}_{cfg.DATA.SPLIT}_{time_string}"
    )

    cfg.TRAIN.LOG_DIR = f"exps/{cfg.TRAIN.SETUP}/{cfg.TRAIN.EXP_NAME}"

    if cfg.TRAIN.DEBUG:
        cfg.TRAIN.LOG_DIR = f"debug/{cfg.TRAIN.LOG_DIR}"
        cfg.TRAIN.NUM_EPOCHS = 1

    cfg.TRAIN.MODEL_DIR = f"{cfg.TRAIN.LOG_DIR}/models"
    cfg.TRAIN.RESULT_DIR = f"{cfg.TRAIN.LOG_DIR}/results"
    os.makedirs(cfg.TRAIN.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.TRAIN.RESULT_DIR, exist_ok=True)

    cfg.TRAIN.LOG_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{cfg.TRAIN.EXP_NAME}.log"
    cfg.TRAIN.RES_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{cfg.TRAIN.EXP_NAME}.txt"
    cfg.TRAIN.CFG_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{cfg.TRAIN.EXP_NAME}.yaml"
    cfg.freeze()
    with open(cfg.TRAIN.CFG_FILENAME, "w") as f:
        f.write(cfg.dump())


def log_redirect(filename):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    update_config(cfg, args)

    # eval
    if cfg.TRAIN.EVAL or cfg.TRAIN.EVAL_CHCKPTS:
        log_redirect(cfg.TRAIN.LOG_FILENAME)

        logging.info(pprint.pformat(args))

        eval(cfg)

    else:  # train
        if not cfg.TRAIN.RESUME:
            extra_train_config(cfg)

        log_redirect(cfg.TRAIN.LOG_FILENAME)

        logging.info(pprint.pformat(args))

        train(cfg)
