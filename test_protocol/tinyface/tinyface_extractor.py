import argparse
import numpy as np

import numpy

from test_protocol.utils.custom_model_loader import load_model

from easydict import EasyDict as edict
from scipy.io import savemat
import torch
import glob
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from data_processor.test_dataset import CommonTestDataset
import logging as logger
logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def extract_id_features(conf):
    dev = 'cuda'
    model = load_model(conf, dev)
    model.use_head = False
    model.use_prev_module = True if conf.module_type else False
    model = torch.nn.DataParallel(model).cuda()

    model.eval()
    folders = ['./distractor_file_list.txt', './probe_file_list.txt', './gallery_file_list.txt']
    batch_size = 256
    for file in folders:
        my_dataset = CommonTestDataset(conf.data_root, file, crop_eye=False)
        data_loader = DataLoader(my_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
        ret_mat = numpy.zeros((len(my_dataset), conf.feat_dim), dtype=float)
        with torch.no_grad():
            for batch_idx, (images, label) in enumerate(data_loader):
                images = images.to(conf.device)
                features = model(images)
                features = F.normalize(features)
                features = features.cpu().numpy()
                ret_mat[batch_idx * batch_size:batch_idx * batch_size + features.shape[0], :] = features
                logger.info('Finished batches: %d/%d.' % (batch_idx+1, len(data_loader)))
        savemat(f"{file.split('/')[1].split('_')[0]}.mat",{f"{file.split('/')[1].split('_')[0]}_feature_map":ret_mat})


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
    conf.add_argument("--data_root", type=str,
                      help="The root folder of testing set.")

    args = conf.parse_args()

    extract_id_features(args)
