import os
import argparse
import logging
import sys
import datetime
import time
import pprint
import yaml
import torch
import numpy as np

from batch_gen import BatchGenerator
from model.mstcn import MSTCNTrainer
from model.asformer import ASFormerTrainer
from model.diffusion import DiffusionTrainer
from utils.misc import seconds_to_hours, set_seed, read_actions
from config import cfg, update_config


def get_trainer(cfg):
    if cfg.MODEL.NAME == "mstcn":
        return MSTCNTrainer(
            cfg.MODEL.PARAMS.NUM_STAGES,
            cfg.MODEL.PARAMS.NUM_LAYERS,
            cfg.MODEL.PARAMS.NUM_F_MAPS,
            cfg.DATA.FEATURE_DIM,
            cfg.DATA.NUM_CLASSES,
        )
    elif cfg.MODEL.NAME == "asformer":
        return ASFormerTrainer(
            cfg.MODEL.PARAMS.NUM_DECODERS,
            cfg.MODEL.PARAMS.NUM_LAYERS,
            cfg.MODEL.PARAMS.R1,
            cfg.MODEL.PARAMS.R2,
            cfg.MODEL.PARAMS.NUM_F_MAPS,
            cfg.DATA.FEATURE_DIM,
            cfg.DATA.NUM_CLASSES,
            cfg.MODEL.PARAMS.CHANNEL_MASK_RATE,
        )
    elif cfg.MODEL.NAME == "diffusion":
        return DiffusionTrainer(
            cfg.MODEL.PARAMS.ENCODER_PARAMS,
            cfg.MODEL.PARAMS.DECODER_PARMS,
            cfg.MODEL.PARAMS.DIFFUSION_PARAMS,
            cfg.DATA.NUM_CLASSES,
            cfg,
        )


def train(cfg):
    trainer = get_trainer(cfg)
    batch_gen = BatchGenerator(
        cfg.DATA.NUM_CLASSES,
        cfg.DATA.ACTIONS_DICT,
        cfg.DATA.GT_PATH,
        cfg.DATA.FEATURES_PATH,
        cfg.DATA.SAMPLE_RATE,
    )
    batch_gen.read_data(cfg.DATA.VID_LIST_FILE)

    start_time = time.time()
    logging.info(f"Starting time: {start_time}")
    trainer.train(
        batch_gen, cfg.TRAIN.MODEL_DIR, cfg.TRAIN.NUM_EPOCHS, cfg.TRAIN.BZ, cfg.TRAIN.LR
    )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg.defrost()
    update_config(cfg, args)

    # eval
    if cfg.TRAIN.EVAL:
        assert (
            cfg.TRAIN.EXP_NAME != ""
        ), "Please provide the correct exp_name for evalutaion"
        exp_name = cfg.TRAIN.EXP_NAME
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=cfg.TRAIN.LOG_FILENAME),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(pprint.pformat(args))
        logging.info("----- evaluation -----")
        # logging.info(cfg)
        eval(cfg)

    else:  # train
        cfg.DATA.VID_LIST_FILE = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/splits/train.split{cfg.DATA.SPLIT}.bundle"
        cfg.DATA.VID_LIST_FILE_TEST = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/splits/test.split{cfg.DATA.SPLIT}.bundle"
        cfg.DATA.FEATURES_PATH = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/features/"
        cfg.DATA.GT_PATH = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/groundTruth/"
        cfg.DATA.MAPPING_FILE = f"{cfg.DATA.PATH}/{cfg.DATA.DATASET}/mapping.txt"
        cfg.DATA.ACTIONS_DICT, cfg.DATA.NUM_CLASSES = read_actions(
            cfg.DATA.MAPPING_FILE
        )

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
        with open(cfg.TRAIN.CONF_FILENAME, "w") as f:
            f.write(cfg.dump())

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=cfg.TRAIN.LOG_FILENAME),
                logging.StreamHandler(sys.stdout),
            ],
        )

        logging.info(pprint.pformat(args))
        # logging.info(cfg)
        logging.info("----- traning -----")
        cfg.freeze()
        set_seed(cfg.TRAIN.SEED)
        train(cfg)
