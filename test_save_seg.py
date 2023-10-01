import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as Grad
from torchvision import transforms
import skimage.io
import os
import copy
from collections import OrderedDict
from tqdm import tqdm, trange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
#from utils.experiment import load_Q_mat
from utils_se import save_imgs,save_imgs_disp

from datasets import KITTI2015loader as kt2015
from datasets import KITTI2012loader as kt2012
from models.MCF_SMSIS import MCF_Stereo_feature_ex,MCF_Stereo_depth,UNetHead
import cv2

def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    print(imgs.shape)
    imgs = imgs.numpy()
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    #for i in range(imgs.shape[0]):
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255
    return imgs_normalized

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default=r"F:\data3\fuqiangjin\gonggonshuju\kitt_fuqiang25\testing\\", help='data path')
parser.add_argument('--kitti', type=str, default='2015')
parser.add_argument('--loadckpt_fe', default='checkpoints/S3-sgdp/checkpoint_000599_fe.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_fes', default='checkpoints/S3-dpsg/fe.pth',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_de', default='checkpoints/S3-sgdp/checkpoint_000599_de.ckpt',help='load the weights from a specific checkpoint')
parser.add_argument('--loadckpt_se', default='checkpoints/S3-dpsg/se.pth',help='load the weights from a specific checkpoint')
args = parser.parse_args()


if args.kitti == '2015':
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath)
else:
    all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2012.kt2012_loader(args.datapath)

test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
test_ldisp = all_ldisp + test_ldisp

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
fe_model.eval()
fes_model.eval()
de_model.eval()
se_model.eval()

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

pred_mae = 0
pred_op = 0
for i in trange(len(test_limg)):
    limg = Image.open(test_limg[i]).convert('RGB')
    rimg = Image.open(test_rimg[i]).convert('RGB')

    w, h = limg.size
    m = 32
    wi, hi = (w // m + 1) * m, (h // m + 1) * m
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    transformsg = transforms.Compose([
        transforms.ToTensor()])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    limg_tensorsg = transformsg(limg)
    limg_tensorsg = dataset_normalized(limg_tensorsg)
    limg_tensorsg = transformsg(limg_tensorsg)
    rimg_tensorsg = transformsg(rimg)
    rimg_tensorsg = dataset_normalized(rimg_tensorsg)
    rimg_tensorsg = transformsg(rimg_tensorsg)
    limg_tensorsg = limg_tensorsg.unsqueeze(0).cuda()
    rimg_tensorsg = rimg_tensorsg.unsqueeze(0).cuda()
    limg_tensorsg = limg_tensorsg.permute(0, 2, 3, 1)
    rimg_tensorsg = rimg_tensorsg.permute(0, 2, 3, 1)
    #print(limg_tensorsg.shape)


    limg_tensor = transform(limg)
    rimg_tensor = transform(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()
    #print(limg_tensor.shape)

    disp_gt = Image.open(test_ldisp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32) / 256
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y = fe_model(limg_tensor, rimg_tensor)
        features_lefts, features_rights, stem_2xs, stem_4xs, stem_2ys, stem_4ys = fes_model(limg_tensorsg, rimg_tensorsg)
        out, sq32, sq16, sq8 = se_model(features_lefts)
        pred_disp = de_model(features_left, features_right, stem_2x, stem_4x, stem_2y, stem_4y,sq32)[-1]
        #pred_disp  = model(limg_tensor, rimg_tensor)[-1]
        pred_disp = pred_disp[:, hi - h:, wi - w:]

        out = out[:, :, hi - h:, wi - w:]
        #print(sq32.shape)
        #print(sq16.shape)
        #print(sq8.shape)

        limg_tensorsg = limg_tensorsg[:, :, hi - h:, wi - w:]

        sq8 = sq8[:, :, :128, :160]
        out = out.squeeze(1).cpu().detach().numpy()
        #limg_tensor = limg_tensor.cpu().detach().numpy()
        save_imgs_disp(limg_tensorsg, sq8, out, i, "seg_pred/", datasets="fuqiang", threshold=0.5, test_data_name=None)
        #print(out)

    predict_np = pred_disp.squeeze().cpu().numpy()
    out = np.where(np.squeeze(out, axis=0) > 0.5, 1, 0)



    op_thresh = 3
    print(disp_gt.shape)
    mask = (disp_gt > 0) & (disp_gt < args.maxdisp)

    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum((pred_error > op_thresh)) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    # print("#### >3.0", np.sum((pred_error > op_thresh)) / np.sum(mask))
    # print("#### EPE", np.mean(pred_error[mask]))

print("#### EPE", pred_mae / len(test_limg))
print("#### >3.0", pred_op / len(test_limg))

