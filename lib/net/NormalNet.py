# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from lib.net.FBNet import define_G
from lib.net.net_util import init_net, VGGLoss
from lib.net.HGFilters import *
from lib.net.BasePIFuNet import BasePIFuNet
import torch
import torch.nn as nn
from lib.net.EMSA import EMSA


class NormalNet(BasePIFuNet):
    '''
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    '''
    def __init__(self, cfg, error_term=nn.SmoothL1Loss()):
        # 调用父类构造函数，初始化 error_term
        super(NormalNet, self).__init__(error_term=error_term)
        # 定义 SmoothL1Loss 损失函数
        self.l1_loss = nn.SmoothL1Loss()
        # 获取配置中的网络参数
        self.opt = cfg.net
        # 如果处于训练模式，则初始化 VGGLoss
        if self.training:
            self.vgg_loss = [VGGLoss()]
        # 过滤并获取正面和背面的输入特征
        self.in_nmlF = [
            item[0] for item in self.opt.in_nml if '_F' in item[0] or item[0] == 'image'
        ]
        self.in_nmlB = [
            item[0] for item in self.opt.in_nml if '_B' in item[0] or item[0] == 'image'
        ]
        # 计算正面和背面的输入维度
        self.in_nmlF_dim = sum(
            [item[1] for item in self.opt.in_nml if '_F' in item[0] or item[0] == 'image']
        )
        self.in_nmlB_dim = sum(
            [item[1] for item in self.opt.in_nml if '_B' in item[0] or item[0] == 'image']
        )
        # 定义正面和背面的生成网络
        self.netF = define_G(self.in_nmlF_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(self.in_nmlB_dim, 3, 64, "global", 4, 9, 1, 3, "instance")
        # 初始化EMSA模块
        self.emsa = EMSA(channels=3)
        # 初始化网络
        init_net(self)

    def forward(self, in_tensor):

        inF_list = []
        inB_list = []
        # 收集正面输入特征
        for name in self.in_nmlF:
            inF_list.append(in_tensor[name])
        # 收集背面输入特征
        for name in self.in_nmlB:
            inB_list.append(in_tensor[name])
        # 正面/背面特征拼接并通过生成网络
        nmlF = self.netF(torch.cat(inF_list, dim=1))
        nmlB = self.netB(torch.cat(inB_list, dim=1))

        # 使用EMSA注意力机制处理特征图
        nmlF = self.emsa(nmlF)
        nmlB = self.emsa(nmlB)

        # ||normal|| == 1，归一化正面和背面特征
        nmlF = nmlF / torch.norm(nmlF, dim=1, keepdim=True)
        nmlB = nmlB / torch.norm(nmlB, dim=1, keepdim=True)

        # output: float_arr [-1,1] with [B, C, H, W]
        # 获取图像掩码
        mask = (in_tensor['image'].abs().sum(dim=1, keepdim=True) != 0.0).detach().float()
        # 应用掩码
        nmlF = nmlF * mask
        nmlB = nmlB * mask

        # 返回正面和背面特征
        return nmlF, nmlB

    def get_norm_error(self, prd_F, prd_B, tgt):
        """calculate normal loss

        Args:
            pred (torch.tensor): [B, 6, 512, 512]
            tagt (torch.tensor): [B, 6, 512, 512]
        """
        # tgt_F, tgt_B = tgt['normal_F'], tgt['normal_B']
        #
        # l1_F_loss = self.l1_loss(prd_F, tgt_F)
        # l1_B_loss = self.l1_loss(prd_B, tgt_B)
        #
        # with torch.no_grad():
        #     vgg_F_loss = self.vgg_loss[0](prd_F, tgt_F)
        #     vgg_B_loss = self.vgg_loss[0](prd_B, tgt_B)
        #
        # total_loss = [5.0 * l1_F_loss + vgg_F_loss, 5.0 * l1_B_loss + vgg_B_loss]
        #
        # return total_loss
        # 获取目标正面和背面特征
        tgt_F, tgt_B = tgt['normal_F'], tgt['normal_B']
        # 计算正面/背面特征的L1损失
        l1_F_loss = self.l1_loss(prd_F, tgt_F)
        l1_B_loss = self.l1_loss(prd_B, tgt_B)
        # 计算正面/背面特征的VGG损失
        with torch.no_grad():
            vgg_F_loss = self.vgg_loss[0](prd_F, tgt_F)
            vgg_B_loss = self.vgg_loss[0](prd_B, tgt_B)

        # Add Laplacian or Total Variation regularization
        tv_F_loss = total_variation_regularization(prd_F)
        tv_B_loss = total_variation_regularization(prd_B)

        # 正则化项的权重
        lambda_reg = 0.1

        # 组合SmoothL1损失、VGG感知损失和Total Variation正则化损失（前法线）
        total_loss_F = 5.0 * l1_F_loss + vgg_F_loss + lambda_reg * tv_F_loss
        # 组合SmoothL1损失、VGG感知损失和Total Variation正则化损失（后法线）
        total_loss_B = 5.0 * l1_B_loss + vgg_B_loss + lambda_reg * tv_B_loss

        # 总损失为L1损失和VGG损失的加权和
        # total_loss = [5.0 * l1_F_loss + vgg_F_loss, 5.0 * l1_B_loss + vgg_B_loss]
        total_loss = [total_loss_F, total_loss_B]
        return total_loss

def total_variation_regularization(normals):
    """Calculate Total Variation regularization for normals"""
    tv_h = torch.mean(torch.abs(normals[:, :, 1:, :] - normals[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(normals[:, :, :, 1:] - normals[:, :, :, :-1]))
    return tv_h + tv_w