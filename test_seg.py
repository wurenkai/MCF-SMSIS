import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *


from engine import *
import os
import sys
from models.MCF_SMSIS import MCF_Stereo_feature_ex,UNetHead
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join('MALUner_latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    





    print('#----------Preparing dataset----------#')
    data_path     = r''
    test_dataset  = isic_loader(path_Data = data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Models----------#')
    fe_model = MCF_Stereo_feature_ex()
    se_model = UNetHead()

    fe_model = torch.nn.DataParallel(fe_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    se_model = torch.nn.DataParallel(se_model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])



    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion





    print('#----------Testing----------#')
    best_weight_fe = torch.load("")
    best_weight_se = torch.load("")
    fe_model.load_state_dict(best_weight_fe['model'])
    se_model.module.load_state_dict(best_weight_se['model_state_dict'])
    loss = test_one_epoch(
        test_loader,
        fe_model,
        se_model,
        criterion,
        logger,
        config,
    )



if __name__ == '__main__':
    config = setting_config
    main(config)