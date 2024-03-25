import random
import torch
import numpy as np
import os
import datetime


def seconds_to_hours(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hours, minutes, seconds


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def create_directories(log_dir, debug):
    if debug:
        log_dir = os.path.join('debug', log_dir)
    model_dir = os.path.join(log_dir, 'models')
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return model_dir, results_dir


def read_actions(mapping_file):
    with open(mapping_file, 'r') as f:
        actions = f.read().splitlines()
    actions_dict = {}
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    return tuple(actions_dict.items()), len(actions_dict)


def generate_exp_name(args):
    time_string = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    exp = f"cd{args.condense}"
    return f"{time_string}_{args.dataset}_{args.model}_{args.split}_{exp}"
