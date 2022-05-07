import argparse
import tqdm
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import (get_env_info, get_root_logger, get_time_str, init_tb_logger,
                           mkdir_and_rename, make_exp_dirs, init_wandb_logger, set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--pretrained_weights', type=str, help='Path to the pretrained weights')
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['pretrained_weights'] = args.pretrained_weights
    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_val_dataloader(opt, logger):
    dataset_opt = opt['datasets']['val']
    val_set = create_dataset(dataset_opt)
    val_loader = create_dataloader(
        val_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])
    logger.info(
        f'Number of val images/folders in {dataset_opt["name"]}: '
        f'{len(val_set)}')
    return val_loader


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
      mkdir_and_rename(osp.join('tb_logger', opt['name']))
    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    val_loader = create_val_dataloader(opt, logger)
    
    # create model
    model = create_model(opt)
    checkpoint = torch.load(opt['pretrained_weights'])
    model.net_g.load_state_dict(checkpoint['params'])

    model.validation(val_loader, 0, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    main()
