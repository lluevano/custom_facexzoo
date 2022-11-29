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

sys.path.append('..')

from bob_test_protocol import score_bob_model, measure_bob_scores
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from modules.module_def import ModuleFactory
from training_mode.conventional_training.train import FaceModel

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
import math

import pickle
import glob


def run_verification(model, cur_epoch, conf, best_eval_criterion, extra_attrs, dask_client=None, groups=["dev",], device=None):
    #  Add verification using bob
    eval_model = model
    eval_model.use_head = False
    #logging.info("Looking for "+os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv"))
    #logging.info(os.path.isfile(os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv")))
    if os.path.isfile(os.path.join(conf.out_dir,f"scores-{groups[0]}-{cur_epoch}.csv")):
        logging.info(f"File scores-{groups[0]}-{cur_epoch}.csv found. Skipping bob pipeline computation")
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

def create_dask_client():
    #from bob.pipelines.config.distributed import sge_io_big
    from dask.distributed import Client

    from bob.pipelines.distributed.sge import SGEMultipleQueuesCluster, get_max_jobs
    from bob.pipelines.distributed.sge_queues import QUEUE_DEFAULT, QUEUE_IOBIG, QUEUE_GPU, QUEUE_MTH
    queue = QUEUE_MTH
    queue["default"]["memory"] = "24GB"
    queue["default"]["io_big"]: False
    queue["default"]["job_extra"]: ["-pe pe_mth 8"]
    min_jobs = 1
    max_jobs = 32# get_max_jobs(queue)
    cluster = SGEMultipleQueuesCluster(min_jobs=min_jobs, sge_job_spec=queue, project="scbiometrics")
    cluster.scale(max_jobs)
    # Adapting to minimim 1 job to maximum 48 jobs
    # interval: Milliseconds between checks from the scheduler
    # wait_count: Number of consecutive times that a worker should be suggested for
    #             removal before we remove it.
    cluster.adapt(
        minimum=min_jobs,
        maximum=max_jobs,
        wait_count=5,
        interval=10,
        target_duration="10s",
    )
    dask_client = Client(cluster)

    return dask_client

def load_model(conf):
    
    pretrain_model = None
    dask_client = create_dask_client()
    conf.device = torch.device('cpu')
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)    
    head_factory = None #HeadFactory('ArcFace', conf.head_conf_file)
    module_factory = ModuleFactory(conf.module_type, conf.module_conf_file) if conf.module_type else None
    model = FaceModel(backbone_factory, head_factory, module_factory)

    dir = os.path.join(conf.out_dir,'*.pt')
    checkpoint_iter = glob.iglob(dir)

    #Decode checkpoint path
    idx = -2 if conf.out_dir.endswith("/") else -1
    folder_name = conf.out_dir.split("/")[idx]
    #ArcFace_nf9_lr_0.0001
    split_folder_name = folder_name.split('_')
    loss = split_folder_name[0]
    nf = split_folder_name[1][2:]
    lr = split_folder_name[3]
    
    extra_attrs={'lr':lr, 'n_unfrozen_layers':nf, 'head_type':loss}

    best_eval_criterion={'EER': math.inf}
    
    for checkpoint_path in checkpoint_iter:
        epoch = checkpoint_path.split("/")[-1].split("_")[-1][:-3]

        try:
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        except KeyError: #  format is likely not to come from facexzoo
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            assert type(state_dict) == dict
            new_state_dict = {}
            for k, v in state_dict.items():
                if k != 'head.weight':
                    new_k_prefix = "" if k.startswith("backbone.") else "backbone."
                    new_state_dict[new_k_prefix+k] = v
            state_dict = new_state_dict
        if 'head.weight' in state_dict.keys():
            #print(state_dict['head.weight'])
            del state_dict['head.weight']
        logger.info(model.load_state_dict(state_dict, strict=False))
        #logger.info(model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']))
        _ = run_verification(model, epoch, conf, best_eval_criterion, extra_attrs, dask_client=dask_client, groups=["dev",], device=conf.device)
        try:
            dask_client.restart()
        except:
            dask_client=create_dask_client()
    
    logger.info("Running final verification on eval set for best EER")
    # load best model
    best_epoch = best_eval_criterion['epoch']
    best_saved_name = f"Epoch_{best_epoch}.pt"
    state_dict = torch.load(os.path.join(conf.out_dir, best_saved_name),map_location=torch.device('cpu'))['state_dict']
    logger.info(model.load_state_dict(state_dict, strict=False))
    logger.info(f"The best dev EER score was {best_eval_criterion['EER']} at epoch {best_eval_criterion['epoch']}")
    scores = run_verification(model, best_epoch, conf, best_eval_criterion, extra_attrs, dask_client=dask_client, groups=["eval",], device=conf.device)
    logger.info("End verification")
    logger.info(f"The eval scores for the best model found are {scores}")

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--backbone_type", type = str,
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, default='../training_mode/backbone_conf.yaml',
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str,
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, default='../training_mode/head_conf.yaml',
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--module_type', default=None,
                      help='Specify the pre-processing module')
    conf.add_argument('--module_conf_file', type=str, default='../training_mode/module_conf.yaml',
                      help="the path of module_conf.yaml.")
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    #conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
    #                  help = 'The path of pretrained model')
    conf.add_argument('--eval_set', type=str, default=None,
                      help='Specify the bob eval set')

    args = conf.parse_args()

    best_dev_pickle = os.path.isfile(os.path.join(args.out_dir, "best_dev.pickle"))
    best_dev_eval_pickle = os.path.isfile(os.path.join(args.out_dir, "best_dev_eval.pickle")) # previous naming convention
    best_eval_pickle = os.path.isfile(os.path.join(args.out_dir, "best_eval.pickle"))
    if best_dev_pickle and (best_dev_eval_pickle or best_eval_pickle):
        logger.info('best dev and eval pickle files present. Skipping evaluations')
    else:
        load_model(args)
    logger.info('Done!')
