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
            cfg.MSTCN.NUM_STAGES,
            cfg.MSTCN.NUM_LAYERS,
            cfg.MSTCN.NUM_F_MAPS,
            cfg.DATA.FEATURE_DIM,
            cfg.DATA.NUM_CLASSES,
        )
    elif cfg.MODEL.NAME == "asformer":
        return ASFormerTrainer(
            cfg.ASFORMER.NUM_DECODERS,
            cfg.ASFORMER.NUM_LAYERS,
            cfg.ASFORMER.R1,
            cfg.ASFORMER.R2,
            cfg.ASFORMER.NUM_F_MAPS,
            cfg.DATA.FEATURE_DIM,
            cfg.DATA.NUM_CLASSES,
            cfg.ASFORMER.CHANNEL_MASK_RATE,
        )
    elif cfg.MODEL.NAME == "diffusion":
        return DiffusionTrainer(
            cfg.DIFFUSION.ENCODER_PARAMS,
            cfg.DIFFUSION.DECODER_PARMS,
            cfg.DIFFUSION.DIFFUSION_PARAMS,
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

    exp = "naive"

    if not cfg.TRAIN.EVAL:
        time_string = datetime.datetime.now().strftime("%m%d-%H-%M-%S-%f")[:-3]
        exp_name = f"{time_string}_{cfg.DATA.DATASET}_{cfg.MODEL.NAME}_{cfg.DATA.SPLIT}"
    else:
        assert cfg.TRAIN.EXP_NAME != "", "Please provide the exp_name for evalutaion"
        exp_name = cfg.TRAIN.EXP_NAME

    cfg.TRAIN.LOG_DIR = f"exps/{exp}/{exp_name}"

    if cfg.TRAIN.DEBUG:
        cfg.TRAIN.LOG_DIR = f"debug/{cfg.TRAIN.LOG_DIR}"
        cfg.TRAIN.NUM_EPOCHS = 1

    cfg.TRAIN.MODEL_DIR = f"{cfg.TRAIN.LOG_DIR}/models"
    cfg.TRAIN.RESULT_DIR = f"{cfg.TRAIN.LOG_DIR}/results"
    os.makedirs(cfg.TRAIN.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.TRAIN.RESULT_DIR, exist_ok=True)

    cfg.TRAIN.LOG_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{exp_name}.log"
    cfg.TRAIN.RES_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{exp_name}.txt"

    cfg.TRAIN.CONF_FILENAME = f"{cfg.TRAIN.LOG_DIR}/{exp_name}.yaml"
    with open(cfg.TRAIN.CONF_FILENAME, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=cfg.TRAIN.LOG_FILENAME),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(pprint.pformat(args))
    logging.info(cfg)
    cfg.freeze()
    set_seed(cfg.TRAIN.SEED)

    if not cfg.TRAIN.EVAL:
        train(cfg)
    else:
        eval(cfg)
