"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from hmac import new
from typing import List, Type
import logging
from openpoints.models.layers.norm import create_norm
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import furthest_point_sample,gather_operation,knn_point,grouping_operation, \
                            create_convblock2d,create_convblock1d,create_act,create_grouper

class PointPreNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels,1,1))
        self.beta = nn.Parameter(torch.zeros(in_channels,1,1))
        self.eps = 1e-6
        
    def forward(self, points):
        anchor = points[:,:,:,0].unsqueeze(-1)
        std = torch.std(points - anchor, dim=1, keepdim=True, unbiased=False)
        points = (points - anchor) / (std + self.eps)

        return points * self.alpha + self.beta


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_idx

class maxpool(nn.Module):
    def __init__(self,k):
        super().__init__()
        self.pool = nn.MaxPool2d((1,k))
    def forward(self,x):
        return self.pool(x).squeeze(-1)

class SetConv(nn.Module):
    def __init__(self, in_channels,out_channels,nsample,prenorm,skip_conn):
        super().__init__()
        self.skip_conn = skip_conn
        self.norm = PointPreNorm(in_channels) if prenorm else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels,out_channels,1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels,out_channels,1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            maxpool(nsample),
        )

        self.act = nn.ReLU(inplace=True)
        self.maxpool = maxpool(nsample)
        
    def forward(self, features):
        features = self.norm(features)
        features = self.conv(features)
        if self.skip_conn:
            features = self.act(self.conv1(features) + self.maxpool(features))
        else:
            features = self.act(self.conv1(self.act(features)))
        return features

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, nsample,prenorm=True,skip_conn=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.nsample = nsample
        self.conv = nn.Sequential(
            SetConv(in_channels,out_channels,nsample,prenorm,skip_conn),
            nn.Conv1d(out_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,1),
            nn.BatchNorm1d(out_channels),
        ) if in_channels!=out_channels else nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        xyz, features = points
        xyz = xyz.contiguous()
        xyz_flipped = xyz.transpose(1,2).contiguous()
        B, N, C = xyz.shape
        S = N//self.stride
        fps_idx = furthest_point_sample(xyz, S)
        sampled_xyz = gather_operation(xyz_flipped, fps_idx).transpose(1,2).contiguous()
        sampled_features = gather_operation(features, fps_idx)

        idx = knn_point(self.nsample, xyz, sampled_xyz).int()
        grouped_points = grouping_operation(features, idx)

        new_features = self.act(self.conv(grouped_points) + self.skip_conv(sampled_features))

        return (sampled_xyz, new_features)
    

@MODELS.register_module()
class PointAnchorEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 width: int = 64,
                 feature_expansion: List[int] = [2,2,2,2],
                 stride : List[int] = [2,2,2,2],
                 nsamples: int = 24,
                 **kwargs
                 ):
        super().__init__()
        self.strides = stride
        layers = []
        self.embedding = create_convblock1d(in_channels, width,)
        channels = [width]
        last_channel = width
        for i in range(len(feature_expansion)):
            layers.append(PointConv(last_channel, 
                                    last_channel * feature_expansion[i], 
                                    stride[i], 
                                    nsamples,
                                    ))
            last_channel = last_channel * feature_expansion[i]
            channels.append(last_channel)
        self.out_channels = channels[-1]
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        f0 = self.embedding(f0)
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return self.pool(f0).squeeze(-1)

    def forward(self, p0, f0=None):
        return self.forward_cls_feat(p0, f0)


if __name__ == '__main__':
    pass