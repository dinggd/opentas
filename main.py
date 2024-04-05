import argparse
import datetime
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch

from batch_gen import BatchGenerator
from config import cfg, update_config
from model.asformer import ASFormerTrainer
from model.diffusion import DiffusionTrainer
from model.mstcn import MSTCNTrainer
from utils.misc import read_actions, seconds_to_hours, set_seed


def get_trainer(cfg):
    if cfg.MODEL.NAME == "mstcn":
        return MSTCNTrainer(cfg)
    elif cfg.MODEL.NAME == "asformer":
        return ASFormerTrainer(cfg)
    elif cfg.MODEL.NAME == "diffusion":
        return DiffusionTrainer(cfg)


def train(cfg):
    set_seed(cfg.TRAIN.SEED)
    trainer = get_trainer(cfg)

    batch_gen = BatchGenerator(cfg)
    batch_gen.read_data(cfg.DATA.VID_LIST_FILE)

    start_time = time.time()
    logging.info(f"Starting time: {start_time}")
    trainer.train(batch_gen, cfg)

    logging.info("Evaluation:")
    scores = trainer.predict(
        cfg.TRAIN.RESULT_DIR,
        cfg.DATA.FEATURES_PATH,
        cfg.DATA.VID_LIST_FILE_TEST,
        cfg.DATA.ACTIONS_DICT,
        cfg.DATA.SAMPLE_RATE,
        cfg.DATA.GT_PATH,
    )

    logging.info(f"{scores}")
    logging.info("End of evaluation.")

    end_time = time.time()
    logging.info(f"Ending time: {end_time}")
    np.savetxt(cfg.TRAIN.RES_FILENAME, np.array([scores]), fmt="%1.2f")

    cost_time = end_time - start_time
    hrs, mins, secs = seconds_to_hours(cost_time)
    logging.info(f"Total time: {hrs:.0f}h:{mins:.0f}m:{secs:.2f}s")


def eval(cfg):
    trainer = get_trainer(cfg)
    trainer.model.load_state_dict(
        torch.load(f"{cfg.TRAIN.MODEL_DIR}/epoch-{cfg.TRAIN.NUM_EPOCHS}.model")
    )
    scores = trainer.predict(
        cfg.TRAIN.RESULT_DIR,
        cfg.DATA.FEATURES_PATH,
        cfg.DATA.VID_LIST_FILE_TEST,
        cfg.DATA.ACTIONS_DICT,
        cfg.DATA.SAMPLE_RATE,
        cfg.DATA.GT_PATH,
    )
    logging.info(f"{scores}")
    np.savetxt(cfg.TRAIN.RES_FILENAME, np.array([scores]), fmt="%1.2f")


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

    cfg.TRAIN.CONF_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{cfg.TRAIN.EXP_NAME}.yaml"
    cfg.freeze()
    with open(cfg.TRAIN.CONF_FILENAME, "w") as f:
        f.write(cfg.dump())


def logging_config(filename):
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
    if cfg.TRAIN.EVAL:

        logging_config(cfg.TRAIN.LOG_FILENAME)

        logging.info(pprint.pformat(args))
        logging.info("-----start  evaluation -----")
        eval(cfg)
        logging.info("-----end of evaluation -----")

    else:  # train
        if not cfg.TRAIN.RESUME:
            extra_train_config(cfg)

        logging_config(cfg.TRAIN.LOG_FILENAME)

        logging.info(pprint.pformat(args))
        logging.info("----- Traning -----")
        train(cfg)
