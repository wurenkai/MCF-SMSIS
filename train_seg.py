import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.MCF_SMSIS import MCF_Stereo_feature_ex,UNetHead
#from dataset.npy_datasets import NPY_datasets
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils_se import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
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
    train_dataset = isic_loader(path_Data = config.data_path, train = True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data = config.data_path, train = False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)
    test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
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
    optimizer_fe = get_optimizer(config, fe_model)
    optimizer_se = get_optimizer(config, se_model)
    scheduler_fe = get_scheduler(config, optimizer_fe)
    scheduler_se = get_scheduler(config, optimizer_se)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1



    #if os.path.exists(resume_model_fe):
    print('#----------Resume fe_Model and Other params----------#')
    resume_model_fe =''
    checkpoint = torch.load(resume_model_fe, map_location=torch.device('cpu'))
    fe_model.load_state_dict(checkpoint['model'])

    a = False
    if a:
        print('#----------Resume se_Model and Other params----------#')
        resume_model_se =''
        checkpoint = torch.load(resume_model_se, map_location=torch.device('cpu'))
        se_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer_se.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_se.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)



    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            fe_model,
            se_model,
            criterion,
            optimizer_fe,
            optimizer_se,
            scheduler_fe,
            scheduler_se,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        loss = val_one_epoch(
                val_loader,
                fe_model,
                se_model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            torch.save(fe_model.state_dict(), os.path.join(checkpoint_dir, 'best_fe.pth'))
            torch.save(se_model.state_dict(), os.path.join(checkpoint_dir, 'best_se.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': fe_model.state_dict(),
                'optimizer_state_dict': optimizer_fe.state_dict(),
                'scheduler_state_dict': scheduler_fe.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest_fe.pth'))

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': se_model.state_dict(),
                'optimizer_state_dict': optimizer_se.state_dict(),
                'scheduler_state_dict': scheduler_se.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest_se.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best_se.pth')):
        print('#----------Testing_best----------#')
        best_weight_fe = torch.load(config.work_dir + 'checkpoints/best_fe.pth', map_location=torch.device('cpu'))
        best_weight_se = torch.load(config.work_dir + 'checkpoints/best_se.pth', map_location=torch.device('cpu'))
        fe_model.load_state_dict(best_weight_fe)
        se_model.load_state_dict(best_weight_se)
        loss = test_one_epoch(
            test_loader,
            fe_model,
            se_model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best_fe.pth'),
            os.path.join(checkpoint_dir, f'best_fe-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best_se.pth'),
            os.path.join(checkpoint_dir, f'best_se-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )

if __name__ == '__main__':
    config = setting_config
    main(config)