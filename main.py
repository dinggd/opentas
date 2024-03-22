#!/usr/bin/python2.7

import torch
from batch_gen import BatchGenerator
import os
import argparse
import random
from model.mstcn import MSTCNTrainer
from model.asformer import ASFormerTrainer
import numpy as np
import logging, sys
import datetime, time
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='1')
parser.add_argument('--condense', default=2, type=int)
parser.add_argument('--model', default='mstcn', choices=['mstcn','asformer','diffusion'])
parser.add_argument('--debug', action="store_true")
parser.add_argument('--seed', default=1538574472)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# seeding
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)


if args.model == 'mstcn':
    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    bz = 8
    lr = 0.0005
    num_epochs = 50

if args.model == 'asformer':
    num_decoders = 4
    num_layers = 10 # J=9 as reported in paper
    num_f_maps = 64
    channel_mask_rate = 0.3
    r1 = 2 # project to q_dim//r1, k_dim//r1 before passing to attention layer
    r2 = 2 # project to v_dim//r2 before passing to attention layer
    bz = 1
    lr = 0.0001
    num_epochs = 30


sample_rate = 1
features_dim = 2048
if args.dataset == "50salads":
    sample_rate = 2

condense_rate = args.condense * sample_rate

vid_list_file = "./datasets/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./datasets/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./datasets/"+args.dataset+"/features/"
gt_path = "./datasets/"+args.dataset+"/groundTruth/"

mapping_file = "./datasets/"+args.dataset+"/mapping.txt"

exp = f"cd{args.condense}"

time_string = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
exp_name = f"{time_string}_{args.dataset}_{args.model}_{args.split}_{exp}"

log_dir = f'exps/{exp}/{exp_name}'

if args.debug:
    log_dir = f'debug/{log_dir}'
    num_epochs = 1

model_dir = f"{log_dir}/models"
results_dir = f"{log_dir}/results"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


log_filename = f'{log_dir}/{exp_name}.log'
res_filename = f'{log_dir}/{exp_name}.txt'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)s] => %(message)s',
    handlers=[
        logging.FileHandler(filename=log_filename),
        logging.StreamHandler(sys.stdout)
    ],
)


# Log arguments
args_dict = vars(args)
logging.info("------------------Arguments:------------------")
for k,v in sorted(args_dict.items()):
    logging.info(f"    {k}: {v}")
logging.info(f"    log_dir: {log_dir}")
logging.info("----------------------------------------------")

def seconds_to_hours(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hours, minutes, seconds

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)


if args.action == "train":

    if args.model == 'mstcn':
        trainer = MSTCNTrainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
    elif args.model == 'asformer':
        trainer = ASFormerTrainer(num_decoders, num_layers, r1, r2, num_f_maps, 
                               features_dim, num_classes, channel_mask_rate)
    
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, condense_rate)
    
    start_time = time.time()
    logging.info(f'Starting time: {start_time}')
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs,bz,lr,device)

    logging.info(f'Evaluation:')
    scores = trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, gt_path)
    logging.info(f'{scores}')
    logging.info('End of evaluation.')
    end_time = time.time()
    logging.info(f"Ending time: {end_time}")
    np.savetxt(res_filename, np.array([scores]), fmt="%1.2f") 
    cost_time = end_time - start_time
    hrs, mins, secs = seconds_to_hours(cost_time)
    logging.info(f"Total time: {hrs:.0f}h:{mins:.0f}m:{secs:.2f}s")

