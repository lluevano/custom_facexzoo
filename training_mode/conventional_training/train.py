"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
sys.path.append('../../')
from test_protocol.bob_test_protocol import score_bob_model, measure_bob_scores

from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import (ImageDataset, TFRecordDB, ImageDataset_HFR)
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from modules.module_def import ModuleFactory

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
import math
import pickle

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory, module_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.prev_module = module_factory.get_module() if module_factory else None
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head() if head_factory else None
        self.use_head = True
        self.use_prev_module = True #passthrough, basically
    def forward(self, data, label=None, ref=None):
        data = data.type(torch.float)
        preprocessed_data = self.prev_module.forward(data) if self.prev_module and self.use_prev_module else data
        feat = self.backbone.forward(preprocessed_data)
        if ref: # HFR Training. Skipping backbone
            assert (self.head.head_type == 'ContrastiveLoss')
            feat_ref = self.backbone.forward(ref)
            pred = self.head.forward(feat, labels=label, ref_emb=feat_ref, ref_labels=label)
        else:
            pred = self.head.forward(feat, label) if self.use_head else feat
        return pred

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']



def unfreeze_first_n_layers(model, n_layers):
    #freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    def _dfs_unfreeze(mod, unfrozen_n, limit_unfreeze):
        total_unfrozen = unfrozen_n
        mod_gen = mod.children()
        while total_unfrozen < limit_unfreeze:
            try:
                #  iterate through module children
                current_mod = next(mod_gen)
                try:
                    #  determine if it's a root or leaf
                    next(current_mod.children())
                    total_unfrozen = _dfs_unfreeze(current_mod, total_unfrozen, limit_unfreeze) # go deeper
                except StopIteration:
                    #  We're in a leaf node
                    #  print("Leaf: " + str(current_mod) + " visited")
                    if total_unfrozen < limit_unfreeze:
                        print("Leaf: " + str(current_mod) + " unfrozen")
                        for param in current_mod.parameters():
                            param.requires_grad = True
                        total_unfrozen += 1
                    else:
                        return total_unfrozen
            except StopIteration:
                #  finish breath iteration
                return total_unfrozen

        return total_unfrozen

    _dfs_unfreeze(model, 0, n_layers)

def run_verification(model, cur_epoch, conf, best_eval_criterion, extra_attrs, dask_client=None, groups=["dev",], device=None):
    #  Add verification using bob
    eval_model = model
    eval_model.use_head = False
    #logging.info("Looking for "+os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv"))
    #logging.info(os.path.isfile(os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv")))
    if os.path.isfile(os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv")):
        logger.info(f"File scores-{groups[0]}-{cur_epoch}.csv found. Skipping bob pipeline computation")
        scores_dict = measure_bob_scores(os.path.join(conf.out_dir, f"scores-{groups[0]}-{cur_epoch}.csv"))
        print(scores_dict)
        scores = [scores_dict]
    else:
        scores = score_bob_model(eval_model, db_name=conf.eval_set, groups=groups, out_dir=conf.out_dir, epoch=cur_epoch, device=device, dask_client=dask_client)

    if (scores[0]['EER'] < best_eval_criterion['EER']) or (groups[0] == "eval"):
        best_eval_criterion['EER'] = scores[0]['EER']
        best_eval_criterion['epoch'] = cur_epoch
        with open(os.path.join(conf.out_dir,f'best_{groups[0]}.pickle'), 'wb') as handle:
            new_conf = extra_attrs
            for k, v in vars(conf).items():
                if not (k in ('writer', 'device','head_type')):
                    new_conf[k] = v
            pickle.dump({'conf': new_conf, 'scores': scores, 'epoch': cur_epoch}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return scores

def train_one_epoch(data_loader, model, optimizer, criterion, cur_epoch, loss_meter, conf, best_eval_criterion):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images_dl, labels) in enumerate(data_loader):
        #print(len(images_dl))
        #print(type(images_dl))
        if isinstance(images_dl, list):
            images = images_dl[0].to(conf.device)
            images_ref = images_dl[1].to(conf.device)
        else:
            images = images_dl.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        elif conf.head_type == 'MagFace':
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            loss = criterion(outputs, labels) + loss_g
        elif conf.head_type == 'ContrastiveLoss':
            if isinstance(images_dl, tuple): #HFR training
                loss = model.forward(images, labels, ref=images_ref)
            else:
                loss = model.forward(images, labels, ref=None)
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), images.shape[0])
        if batch_idx % conf.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if (batch_idx + 1) % conf.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
    saved_name = 'Epoch_%d.pt' % cur_epoch
    state = {'state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
             'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(conf.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

    #  Add verification using bob
    if conf.eval_set:
        eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        eval_model.use_head = False
        _ = run_verification(eval_model, cur_epoch, conf, best_eval_criterion, {'head_type': conf.head_type}, dask_client=None, groups=["dev",], device=conf.device)
        eval_model.use_head = True
        eval_model.train()
        logger.info('End verification')
def train(conf):
    """Total training procedure.
    """
    if conf.ref_file and not (conf.train_file.endswith('.tfrecord')):
        data_loader = DataLoader(ImageDataset_HFR(conf.data_root, conf.train_file, conf.ref_file),
                                 conf.batch_size, True, num_workers=4)
    elif conf.train_file.endswith('.tfrecord'):  # Adapted code to read tfrecord
        #  It is advisable to provide the db_size beforehand since it can take some time calculating it
        #  MSCeleb-ArcFace (also called msceleb-v2 or Emore). Num samples: 5822653
        tfrecord_iterable_db = TFRecordDB(tfrecord_path=conf.train_file, shuffle_queue_size=1024, HFR=conf.ref_file, db_size=5822653)
        data_loader = DataLoader(tfrecord_iterable_db, batch_size=conf.batch_size)
    else:
        data_loader = DataLoader(ImageDataset(conf.data_root, conf.train_file),
                                 conf.batch_size, True, num_workers=4)
    conf.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device) if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    module_factory = ModuleFactory(conf.module_type, conf.module_conf_file) if conf.module_type else None
    model = FaceModel(backbone_factory, head_factory, module_factory)
    ori_epoch = 0
    if conf.resume:
        ori_epoch = ori_epoch if conf.fine_tune else torch.load(args.pretrain_model, map_location=conf.device)['epoch'] # TODO catch keyerror
        ori_epoch += 1
        try:
            state_dict = torch.load(args.pretrain_model, map_location=conf.device)['state_dict']
        except KeyError: #  format is likely not to come from facexzoo
            state_dict = torch.load(args.pretrain_model, map_location=conf.device)
            assert type(state_dict) == dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k != 'head.weight':
                    new_k_prefix = "" if k.startswith("backbone.") else "backbone."
                    new_state_dict[new_k_prefix+k] = v
            state_dict = new_state_dict
        if conf.fine_tune and ('head.weight' in state_dict.keys()):
            #print(state_dict['head.weight'])
            del state_dict['head.weight']
        logger.info(model.load_state_dict(state_dict, strict=False))
        #  freeze and unfreeze some layers
        if conf.fine_tune and (not (conf.n_unfrozen_layers is None)):
            unfreeze_first_n_layers(model, conf.n_unfrozen_layers) # also affects prev_module
            if conf.module_type: # Unfreezing prev_module
                for name, param in model.named_parameters():
                    if name.startswith("prev_module"):
                        param.requires_grad = True
                        logger.info(f"Unfrozen parameter {name}")

    model = torch.nn.DataParallel(model).cuda() if torch.cuda.is_available() else model

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)
    loss_meter = AverageMeter()
    model.train()
    best_eval_criterion={'EER': math.inf}
    #if conf.eval_set:
    #    logger.info("Starting point on dev set")
    #    eval_model = model.module
    #    eval_model.use_head = False
    #    scores = score_bob_model(eval_model, db_name=conf.eval_set, groups=["dev"], out_dir=conf.out_dir, epoch="start")
    #    logger.info(f"{scores}")
    #    eval_model.use_head = True
    #    eval_model.train()
        #logger.info("End verification")

    for epoch in range(ori_epoch, conf.epoches):
        train_one_epoch(data_loader, model, optimizer, 
                        criterion, epoch, loss_meter, conf, best_eval_criterion)
        lr_schedule.step()
    #  Finished training

    if conf.eval_set:
        logger.info("Running final verification on eval set for best EER")
        # load best model
        best_saved_name = 'Epoch_%d.pt' % best_eval_criterion['epoch']
        state_dict = torch.load(os.path.join(args.out_dir, best_saved_name), map_location=conf.device)['state_dict']
        eval_model = model.module if isinstance(model,torch.nn.DataParallel) else model
        eval_model.load_state_dict(state_dict, strict=False)
        eval_model.use_head = False
        logger.info(f"The best dev EER score was {best_eval_criterion['EER']} at epoch {best_eval_criterion['epoch']}")
        scores = run_verification(eval_model, best_eval_criterion['epoch'], conf, best_eval_criterion, {'head_type': conf.head_type}, dask_client=None, groups=["eval",], device=conf.device)
        logger.info("End verification")
        logger.info(f"The eval scores for the best model found are {scores}")

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str,
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, default='../backbone_conf.yaml',
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str,
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, default='../head_conf.yaml',
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--module_type', default=None,
                      help='Specify the pre-processing module')
    conf.add_argument('--module_conf_file', type=str, default='../module_conf.yaml',
                      help="the path of module_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    conf.add_argument('--fine_tune', '-ft', action = 'store_true', default = False,
                      help='Specify if fine tuning to another dataset.')
    conf.add_argument('--n_unfrozen_layers', type=float, default=None,
                      help='Specify the number of trainable layers')
    conf.add_argument('--eval_set', type=str, default=None,
                      help='Specify the bob eval set')
    conf.add_argument('--ref_file', type=str, default=None,
                      help='Specify a reference file list for heterogeneous FR')

    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
