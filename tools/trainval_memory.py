from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb
from model.train_val_memory import train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from roi_data_layer.roidb import prepare_roidb
import nets.attend_memory as attend_memory
import nets.base_memory as base_memory
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a region classification network with memory')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default='/home/wangwenbin/Method/scene_graph/tf-faster-rcnn/data/imagenet_weights/res50.ckpt',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='visual_genome_train_rel', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        default='visual_genome_val_rel', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50_local', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_name):
    """
    Combine multiple roidbs
    """
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)
    #if cfg.TRAIN.USE_RPN_DB:
    #    roidb = imdb.add_rpn_rois()
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set

    imdb, roidb = combined_roidb(args.imdb_name)
    prepare_roidb(roidb)

    print('Loaded imdb `{:s}` for training'.format(args.imdb_name))
    print('{:d} roidb entries'.format(len(roidb)))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    valimdb, valroidb = combined_roidb(args.imdbval_name)
    prepare_roidb(valroidb)
    cfg.TRAIN.USE_FLIPPED = orgflip
    print('Loaded imdb `{:s}` for validating'.format(args.imdbval_name))
    print('{:d} validation roidb entries'.format(len(valroidb)))

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    net_base, net_tag = args.net.split('_')

    if net_tag == 'local':
        memory = base_memory
    else:
        raise NotImplementedError

    # load network
    if net_base == 'vgg16':
        net = memory.vgg16_memory()
    elif net_base == 'res50':
        net = memory.resnetv1_memory(num_layers=50)
    elif net_base == 'res101':
        net = memory.resnetv1_memory(num_layers=101)
    elif net_base == 'res152':
        net = memory.resnetv1_memory(num_layers=152)
    elif net_base == 'mobile':
        net = memory.mobilenetv1_memory()
    else:
        raise NotImplementedError

    train_net(net, imdb, roidb, valimdb, valroidb, output_dir, tb_dir,
              pretrained_model=args.weight,
              max_iters=args.max_iters)
