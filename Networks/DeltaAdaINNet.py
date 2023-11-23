from __future__ import absolute_import, division, print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch.autograd import Variable

# style_type='VGG' #'resnet'
style_type = "resnet"


def activate_fn(x, inplace=True):
    return F.relu(x, inplace=inplace)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mapping(content_feat, style_st):
    size = content_feat.size()
    if len(size) < 3:
        return mapping2(content_feat, style_st)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean) / (content_std + 1e-8)

    style_mean, style_std = (
        -style_st[1].view(-1, 1, 1, 1) + content_mean,
        style_st[0].view(-1, 1, 1, 1) - content_std,
    )
    style_feat = normalized_feat * torch.abs(style_std) + style_mean
    return style_feat


def mapping2(content_feat, style_st):
    eps = 1e-6
    b = content_feat.size(0)
    content_feat = content_feat.view(b, -1)
    content_mean = content_feat.mean(dim=-1).view(b, 1)
    content_var = content_feat.view(b, -1).var(dim=-1) + eps
    content_std = content_var.sqrt().view(b, 1)
    normalized_feat = (content_feat - content_mean) / (content_std + eps)
    style_mean, style_std = content_mean - style_st[1].view(
        -1, 1
    ), content_std - style_st[0].view(-1, 1)
    style_feat = normalized_feat * torch.abs(style_std) + style_mean
    return style_feat


class DeltaAdaIN(nn.Module):
    def __init__(self, num_classes=100, da_type="binary"):
        super(DeltaAdaIN, self).__init__()
        self.num_classes = num_classes
        self.da_type = da_type

        if self.da_type != "image_template":
            self.latents = self.create_latents()
            self.mean_std_encoder = nn.Sequential(
                nn.Linear(self.latents.size(-1), 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

    def create_latents(self):
        latents = []
        if self.da_type == "binary":
            for i in range(self.num_classes):
                latents.append([int(x) for x in bin(i + 1)[2:].zfill(8)])
            latents = (
                torch.tensor(latents, requires_grad=False)
                .view(self.num_classes, -1)
                .float()
            )
            latents = F.normalize(latents, dim=-1)
        else:
            latents = [i + 1 for i in range(self.num_classes)]
            latents = (
                torch.tensor(latents, requires_grad=False)
                .view(self.num_classes, -1)
                .float()
            )
        return latents

    def calc_mean_std(self, feat, eps=1e-8):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def calc_mean_std2(self, feat, eps=1e-8):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, -1).var(dim=1) + eps
        feat_std = feat_var.sqrt().view(N, 1, 1, 1)
        feat_mean = feat.view(N, -1).mean(dim=1).view(N, 1, 1, 1)
        return feat_mean, feat_std

    def delta_adain_operation(self, style_mean_std, content):
        content_mean, content_std = self.calc_mean_std(content)
        content = (content - content_mean) / content_std

        style_mean, style_std = style_mean_std

        b1, b2 = content_mean.size(0), style_mean.size(0)
        content_mean = (
            content_mean.view(b1, 1, -1).repeat(1, b2, 1).view(b1, b2, -1, 1, 1)
        )
        content_std = (
            content_std.view(b1, 1, -1).repeat(1, b2, 1).view(b1, b2, -1, 1, 1)
        )
        style_mean = style_mean.view(1, b2, -1).repeat(b1, 1, 1).view(b1, b2, -1, 1, 1)
        style_std = style_std.view(1, b2, -1).repeat(b1, 1, 1).view(b1, b2, -1, 1, 1)

        content = content[:, None, ...].repeat(1, b2, 1, 1, 1)
        da_feats = (style_std - content_std) * content + (style_mean - content_mean)
        return da_feats

    def forward(self, x, template_x=None):
        outputs = {}
        eps = 1e-8
        if template_x is not None:
            style_mean, style_std = self.calc_mean_std2(template_x)
        else:
            self.latents = self.latents.to(x)
            style_mean, style_std = torch.split(
                self.mean_std_encoder(self.latents), [1, 1], -1
            )

        da_feats = self.delta_adain_operation([style_mean, style_std], x)
        return da_feats
