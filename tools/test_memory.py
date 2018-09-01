from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
import nets.base_memory as base_memory
import nets.attend_memory as attend_memory
from datasets.factory import get_imdb
from roi_data_layer.roidb import prepare_roidb
import argparse
import pprint
import time, os, sys

import tensorflow as tf


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a region classification network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--visualize', dest='visualize', help='whether to show results',
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='visual_genome_test_rel', type=str)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res50_local', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


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

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    imdb = get_imdb(args.imdb_name)
    roidb = imdb.roidb
    if cfg.TEST.USE_RPN_DB:
        imdb.add_rpn_rois(roidb, make_copy=False)
    prepare_roidb(roidb)

    cfg.CLASSES = imdb.classes

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)

    net_base, net_tag = args.net.split('_')

    if net_tag == 'local':
        memory = base_memory
        iter_test = False
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

    # load model
    net.create_architecture("TEST", imdb.num_classes, imdb.num_predicates, tag='default')

    if args.model:
        print(('Loading model check point from {:s}').format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        print('Loaded.')
    else:
        print(('Loading initial weights from {:s}').format(args.weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded.')

    test_net(sess, net, imdb, roidb, filename, args.visualize, iter_test=iter_test, mode='all')

    sess.close()
