"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import logging
import os
import sys
import shutil
import argparse
import logging as logger

import torch

sys.path.append('../..')

from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from modules.module_def import ModuleFactory
from training_mode.conventional_training.train import FaceModel

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def load_model(conf, device='cpu'):
    conf.device = torch.device(device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)
    head_factory = None  # HeadFactory('ArcFace', conf.head_conf_file)
    module_factory = ModuleFactory(conf.module_type, conf.module_conf_file) if conf.module_type else None
    model = FaceModel(backbone_factory, head_factory, module_factory)

    conf.feat_dim = backbone_factory.backbone_param['feat_dim']
    # Decode checkpoint path
    checkpoint_path = conf.pretrain_model
    try:
        state_dict = torch.load(checkpoint_path, map_location=conf.device)['state_dict']
    except KeyError:  # format is likely not to come from facexzoo
        state_dict = torch.load(checkpoint_path, map_location=conf.device)
        assert type(state_dict) == dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k != 'head.weight':
                new_k_prefix = "" if k.startswith("backbone.") else "backbone."
                new_state_dict[new_k_prefix + k] = v
        state_dict = new_state_dict
    if 'head.weight' in state_dict.keys():
        # print(state_dict['head.weight'])
        del state_dict['head.weight']
    logger.info(model.load_state_dict(state_dict, strict=False))
    return model

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='model loading for face recognition.')
    conf.add_argument("--backbone_type", type=str,
                      help="Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type=str, default='../../training_mode/backbone_conf.yaml',
                      help="the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type=str,
                      help="mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type=str, default='../../training_mode/head_conf.yaml',
                      help="the path of head_conf.yaml.")
    conf.add_argument('--module_type', default=None,
                      help='Specify the pre-processing module')
    conf.add_argument('--module_conf_file', type=str, default='../../training_mode/module_conf.yaml',
                      help="the path of module_conf.yaml.")
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt',
                      help = 'The path of pretrained model')

    args = conf.parse_args()

    load_model(args)
