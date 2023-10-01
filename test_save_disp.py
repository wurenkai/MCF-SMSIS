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
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
from models.MCF_SMSIS import MCF_Stereo_feature_ex,MCF_Stereo_depth,UNetHead
import cv2
# cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath_12', default="/data/KITTI/KITTI_2012/", help='data path')
parser.add_argument('--datapath_15', default=r"F:\data3\fuqiangjin\gonggonshuju\kitt_fuqiang25\\", help='data path')
parser.add_argument('--testlist',default='./filenames/SCARED8_test.txt', help='testing list')
parser.add_argument('--loadckpt_fe', default='checkpoints/S3-sgdp/checkpoint_000599_fe.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_fes', default='checkpoints/S3-dpsg/fe.pth',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_de', default='checkpoints/S3-sgdp/checkpoint_000599_de.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_se', default='checkpoints/S3-dpsg/se.pth',help='load the weights from a specific checkpoint')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath_15, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
fe_model = MCF_Stereo_feature_ex()
fes_model = MCF_Stereo_feature_ex()
de_model = MCF_Stereo_depth()
se_model = UNetHead()
fe_model = nn.DataParallel(fe_model)
fes_model = nn.DataParallel(fes_model)
de_model = nn.DataParallel(de_model)
se_model = nn.DataParallel(se_model)
fe_model.cuda()
fes_model.cuda()
de_model.cuda()
se_model.cuda()



###load parameters
fe_state_dict = torch.load(args.loadckpt_fe)
fe_model.load_state_dict(fe_state_dict['model'])
fes_state_dict = torch.load(args.loadckpt_fes)
fes_model.load_state_dict(fes_state_dict)
#fes_model.module.load_state_dict(fes_state_dict['model_state_dict'])
de_state_dict = torch.load(args.loadckpt_de)
de_model.load_state_dict(de_state_dict['model'])
se_state_dict = torch.load(args.loadckpt_se, map_location=torch.device('cpu'))
#se_model.module.load_state_dict(se_state_dict['model_state_dict'])
se_model.load_state_dict(se_state_dict)


save_dir = './test'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.synchronize()
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        torch.cuda.synchronize()
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
            #assert len(disp_est.shape) == 2
            #disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)

            fn = os.path.join(save_dir, fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            skimage.io.imsave(fn, disp_est_uint)
            # cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01),cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


# test one sample
@make_nograd_func
def test_sample(sample):
    fe_model.eval()
    fes_model.eval()
    de_model.eval()
    se_model.eval()
    features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y = fe_model(sample['left'].cuda(), sample['right'].cuda())
    features_lefts, features_rights, stem_2xs, stem_4xs, stem_2ys, stem_4ys = fes_model(sample['left'].cuda(), sample['right'].cuda())
    out, sq32, sq16, sq8 = se_model(features_lefts)
    disp_ests = de_model(features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y, sq32 = sq32)
    #disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


if __name__ == '__main__':
    test()
