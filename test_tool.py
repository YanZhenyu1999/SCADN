from model.model import Generator, Discriminator
from utils.datasets import Trainset, Testset
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
from torch.optim import Adam
import torch
import os
import yaml
import easydict
import argparse
import logging
from test import test


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='the path of config file', default='./configs/config_mvtec.yml')
parser.add_argument('--checkpoint', type=str, help='model checkpoints path', default='./checkpoints/50.pth')
parser.add_argument('--gpu', type=int, nargs='+', help='gpu list')
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
config = easydict.EasyDict(config)

# if args.gpu is not None:
#     config.GPU = list(args.gpu)
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(id) for id in config.GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

logname = os.path.join(config.LOG_PATH, 'train.log')
log_file = open(logname, 'a')

write_dir = config.WRITE_PATH + 'seed' + str(config.SEED)
if not os.path.exists(write_dir):
    os.makedirs(write_dir)
else:
    shutil.rmtree(write_dir)
    os.mkdir(write_dir)
writer = SummaryWriter(write_dir)


test(config, 1, log_file, writer)
log_file.close()
writer.close()
