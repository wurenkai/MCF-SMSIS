from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
from models.MCF_SMSIS import MCF_Stereo_feature_ex,MCF_Stereo_depth,UNetHead

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
#parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default=r"E:\data2\SCENE_FLOW_ALL\flyingthings3d__frames_finalpass\\", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')

# parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default="/home/xgw/data/KITTI_2015/", help='data path')
# parser.add_argument('--trainlist', default='./filenames/kitti12_15_all.txt', help='training list')
# parser.add_argument('--testlist',default='./filenames/kitti15_all.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="10,14,16,18:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='checkpoints/p1_autolape', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt_fe', default='results/mal_Horunet_fuqiang_device_Friday_28_July_2023_21h_05m_58s/checkpoints/best_fe-epoch161-loss0.1380.pth', help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_de', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=2, drop_last=False)

# model, optimizer
#model = __models__[args.model](args.maxdisp)
fe_model = MCF_Stereo_feature_ex()
de_model = MCF_Stereo_depth()
fe_model = nn.DataParallel(fe_model)
de_model = nn.DataParallel(de_model)
fe_model.cuda()
de_model.cuda()
optimizer_fe = optim.Adam(fe_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
optimizer_de = optim.Adam(de_model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts_fe = [fn for fn in os.listdir(args.logdir) if fn.endswith("_fe.ckpt")]
    all_saved_ckpts_fe = sorted(all_saved_ckpts_fe, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt_fe = os.path.join(args.logdir, all_saved_ckpts_fe[-1])
    print("loading the lastest fe_model in logdir: {}".format(loadckpt_fe))
    state_dict_fe = torch.load(loadckpt_fe)
    fe_model.load_state_dict(state_dict_fe['model'])
    optimizer_fe.load_state_dict(state_dict_fe['optimizer'])
    start_epoch = state_dict_fe['epoch'] + 1

    all_saved_ckpts_de = [fn for fn in os.listdir(args.logdir) if fn.endswith("_de.ckpt")]
    all_saved_ckpts_de = sorted(all_saved_ckpts_de, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt_de = os.path.join(args.logdir, all_saved_ckpts_de[-1])
    print("loading the lastest de_model in logdir: {}".format(loadckpt_de))
    state_dict_de = torch.load(loadckpt_de)
    de_model.load_state_dict(state_dict_de['model'])
    optimizer_de.load_state_dict(state_dict_de['optimizer'])
    start_epoch = state_dict_de['epoch'] + 1

elif args.loadckpt_fe:
    # load the checkpoint file specified by args.loadckpt_fe
    print("loading fe_model {}".format(args.loadckpt_fe))
    state_dict_fe = torch.load(args.loadckpt_fe)
    fe_model_dict = fe_model.state_dict()
    #pre_dict_fe = {k: v for k, v in state_dict_fe['model'].items() if k in fe_model_dict}
    pre_dict_fe = {k: v for k, v in state_dict_fe.items() if k in fe_model_dict}
    fe_model_dict.update(pre_dict_fe)
    # model.load_state_dict(state_dict['model'])
    fe_model.load_state_dict(fe_model_dict)

elif args.loadckpt_de:
    # load the checkpoint file specified by args.loadckpt_de
    print("loading de_model {}".format(args.loadckpt_de))
    state_dict_de = torch.load(args.loadckpt_de)
    de_model_dict = de_model.state_dict()
    pre_dict_de = {k: v for k, v in state_dict_de['model'].items() if k in de_model_dict}
    de_model_dict.update(pre_dict_de)
    # model.load_state_dict(state_dict['model'])
    de_model.load_state_dict(de_model_dict)
print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer_fe, epoch_idx, args.lr, args.lrepochs)
        adjust_learning_rate(optimizer_de, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': fe_model.state_dict(), 'optimizer': optimizer_fe.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}_fe.ckpt".format(args.logdir, epoch_idx))
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': de_model.state_dict(), 'optimizer': optimizer_de.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}_de.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))


# train one sample
def train_sample(sample, compute_metrics=False):
    fe_model.train()
    de_model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()
    disp_gt_low = disp_gt_low.cuda()
    optimizer_fe.zero_grad()
    optimizer_de.zero_grad()

    #disp_ests = model(imgL, imgR)
    features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y = fe_model(imgL, imgR)
    disp_ests = de_model(features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    masks = [mask, mask_low]
    disp_gts = [disp_gt, disp_gt_low] 
    loss = model_loss_train(disp_ests, disp_gts, masks)
    disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
    loss.backward()
    optimizer_fe.step()
    optimizer_de.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    fe_model.eval()
    de_model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    #disp_ests = model(imgL, imgR)
    features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y = fe_model(imgL, imgR)
    disp_ests = de_model(features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
