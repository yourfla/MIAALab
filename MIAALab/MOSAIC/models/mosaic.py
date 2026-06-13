"""
MOSAIC: Multi-rater Opinion Segmentation with Annotator-Informed Calibration.

This module implements the MOSAIC model, a unified multi-rater segmentation
framework built on three gradient-isolated modules:

  - SC-ECRD : Style-Conditioned Expert-Aware Conditional Refinement Diffusion,
              which injects annotator-specific directional offsets into the
              latent prior to produce attributable, calibrated diversity.
  - EBF     : Evidential Belief Fusion, a style-conditioned Dirichlet evidence
              pathway whose differentiable weighted belief fusion yields a
              pixel-level decomposition of inherent and inter-rater uncertainty
              together with per-annotator probability maps.
  - SABR    : Spatially-Aware Boundary Refinement, which gates a lightweight
              corrective module to the contested boundary pixels localized by
              EBF's inter-rater uncertainty.

The model uses a two-stage training protocol: a base stage that trains the
U-Net, prior/posterior encoders and segmentation head, followed by an auxiliary
stage that trains the three modules above with the base network frozen.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from Probabilistic_Unet_Pytorch.unet_blocks import *
from Probabilistic_Unet_Pytorch.unet import Unet
from Probabilistic_Unet_Pytorch.utils import (
    init_weights, init_weights_orthogonal_normal, l2_regularisation
)
from torch.distributions import Normal, Independent, kl
from pionono_models.model_headless import UnetHeadless

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# ============================================================

class Conv_block(nn.Module):
    def __init__(self, input_channels, num_filters, padding=True):
        super(Conv_block, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(0.1))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        return self.layers(input)

class Encoder(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 initializers, padding=True, posterior=False, num_annotators=4):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        if posterior:
            self.input_channels += num_annotators
        layers = []
        for i in range(len(self.num_filters)):
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        return self.layers(input)

class AxisAlignedConvGaussian(nn.Module):
    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 latent_dim, initializers, posterior=False, num_annotators=4):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.name = 'Posterior' if self.posterior else 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters,
                               self.no_convs_per_block, initializers,
                               posterior=self.posterior,
                               num_annotators=num_annotators)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1, 1), stride=1)
        self.show_img = 0; self.show_seg = 0; self.show_concat = 0
        self.show_enc = 0; self.sum_input = 0
        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):
        if segm is not None:
            self.show_img = input; self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input; self.sum_input = torch.sum(input)
        encoding = self.encoder(input)
        self.show_enc = encoding
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        return Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

class Projection(nn.Module):
    def __init__(self, num_experts, latent_dim, style_dim=32):
        super(Projection, self).__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.multi_expert_heads = nn.ModuleList([
            Conv_block(16, [32, 16, 6], 3) for _ in range(self.num_experts)])
        self.pooling_layer = nn.AdaptiveAvgPool2d([1, 1])
        self.activation = torch.nn.Softmax(dim=2)
        out_ch = 6
        self.style_film = nn.Sequential(
            nn.Linear(style_dim, style_dim), nn.GELU(),
            nn.Linear(style_dim, out_ch * 2))
        nn.init.zeros_(self.style_film[-1].weight)
        nn.init.zeros_(self.style_film[-1].bias)

    def forward(self, feature_map, z_set, idx, expert_style=None):
        feats = self.multi_expert_heads[idx](feature_map)
        if expert_style is not None:
            film_params = self.style_film(expert_style)
            scale, shift = film_params.chunk(2, dim=1)
            feats = feats * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        bs = feature_map.shape[0]
        global_z = self.pooling_layer(feats).view(bs, self.latent_dim, -1).permute(0, 2, 1)
        similarity = torch.bmm(global_z, z_set.permute(0, 2, 1))
        similarity = self.activation(similarity)
        return torch.bmm(similarity, z_set)

class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid())
        nn.init.zeros_(self.fc[-2].weight)
        nn.init.constant_(self.fc[-2].bias, 0.0)

    def forward(self, x):
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)
        return x * self.fc(w).view(B, C, 1, 1)

class EvidentialFcomb(nn.Module):
    """

    Forward path:
    """
    def __init__(self, num_filters, latent_dim, num_output_channels,
                 num_classes, no_convs_fcomb, initializers, style_dim=32,
                 use_tile=True):
        super(EvidentialFcomb, self).__init__()
        self.num_channels = num_output_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'
        self.style_dim = style_dim
        self.feat_ch = self.num_filters[0]
        if self.use_tile:
            self.conv1 = nn.Conv2d(16 + self.latent_dim, self.feat_ch, kernel_size=1)
            mid_layers = []
            for _ in range(no_convs_fcomb - 2):
                mid_layers.append(nn.Conv2d(self.feat_ch, self.feat_ch, kernel_size=1))
                mid_layers.append(nn.ReLU(inplace=True))
            self.mid_layers = nn.Sequential(*mid_layers)
            self.last_layer = nn.Conv2d(self.feat_ch, self.num_classes, kernel_size=1)

            self.film_proj1 = nn.Linear(style_dim, style_dim)
            self.film_act = nn.GELU()
            self.film_proj2 = nn.Linear(style_dim, self.feat_ch * 2)
            nn.init.zeros_(self.film_proj2.weight)
            nn.init.zeros_(self.film_proj2.bias)
            self.se = ChannelSE(self.feat_ch, reduction=4)
            self.film2_style_proj = nn.Linear(style_dim, style_dim)
            self.film2_act = nn.GELU()
            self.film2_proj = nn.Linear(style_dim, self.feat_ch * 2)
            nn.init.zeros_(self.film2_proj.weight)
            nn.init.zeros_(self.film2_proj.bias)
            self.evi_last_layer = nn.Conv2d(self.feat_ch, 2, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.conv1.apply(init_weights_orthogonal_normal)
                self.mid_layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
                self.evi_last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.conv1.apply(init_weights)
                self.mid_layers.apply(init_weights)
                self.last_layer.apply(init_weights)
                self.evi_last_layer.apply(init_weights)
            with torch.no_grad():
                self.evi_last_layer.bias[0].fill_(-0.5)
                self.evi_last_layer.bias[1].fill_(0.5)
        self.activation = torch.nn.Softmax(dim=1)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(device)
        return torch.index_select(a, dim, order_index)

    def _apply_film(self, h, film_params):
        scale, shift = film_params.chunk(2, dim=1)
        return h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, feature_map, z, expert_style=None, use_softmax=True,
                return_evidence=False, detach_base=False, use_logit_head=False):
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            h = self.conv1(feature_map)
            if detach_base:
                h = h.detach()
            h = F.relu(h, inplace=False)
            h_mid = self.mid_layers(h)
            if detach_base:
                h_mid = h_mid.detach()

            if expert_style is None and not return_evidence and not use_logit_head:
                raw_out = self.last_layer(h_mid)
                if use_softmax:
                    raw_out = torch.sigmoid(raw_out)
                return raw_out

            if use_logit_head:
                if expert_style is not None:
                    style_feat1 = self.film_act(self.film_proj1(expert_style))
                    h_styled = self._apply_film(h, self.film_proj2(style_feat1))
                    h_styled = self.se(h_styled)
                    h_styled = F.relu(h_styled, inplace=False)
                    h_mid_styled = self.mid_layers(h_styled)
                    return self.last_layer(h_mid_styled)
                else:
                    return self.last_layer(h_mid)

            # h_mid -> [FiLM1-SE-FiLM2] -> evi_last_layer -> softplus
            h_evi = h_mid
            if expert_style is not None:
                style_feat1 = self.film_act(self.film_proj1(expert_style))
                h_evi = self._apply_film(h_evi, self.film_proj2(style_feat1))
                h_evi = self.se(h_evi)
                style_feat2 = self.film2_act(self.film2_style_proj(expert_style))
                h_evi = self._apply_film(h_evi, self.film2_proj(style_feat2))
            return F.softplus(self.evi_last_layer(h_evi))

class ExpertStyleBank(nn.Module):
    def __init__(self, num_experts=4, style_dim=32):
        super().__init__()
        self.num_experts = num_experts
        self.style_dim = style_dim
        self.embeddings = nn.Embedding(num_experts, style_dim)
        nn.init.normal_(self.embeddings.weight, std=1.0)

    def forward(self, expert_idx):
        return self.embeddings(expert_idx)

    def mean_style(self):
        return self.embeddings.weight.mean(dim=0)

    def orthogonality_loss(self):
        W = F.normalize(self.embeddings.weight, dim=1)
        gram = W @ W.T
        identity = torch.eye(self.num_experts, device=W.device)
        return ((gram - identity) ** 2).sum() / (self.num_experts * (self.num_experts - 1))

class ECRDStyleProjector(nn.Module):
    def __init__(self, style_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(style_dim, style_dim), nn.GELU(),
            nn.Linear(style_dim, style_dim))
        nn.init.eye_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)
        nn.init.eye_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, style):
        return self.proj(style)

class DifferentiableWBF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, evidences):
        opinions = []
        for evi in evidences:
            alpha = evi + 1
            S = alpha.sum(dim=1, keepdim=True)
            opinions.append({
                'belief': evi[:, 0:1] / S,
                'uncertainty': 2.0 / S,
                'confidence': (1.0 - 2.0 / S).clamp(min=1e-8),
                'alpha': alpha, 'S': S})
        confidences = torch.cat([o['confidence'] for o in opinions], dim=1)
        weights = confidences / (confidences.sum(dim=1, keepdim=True) + 1e-8)
        beliefs = torch.cat([o['belief'] for o in opinions], dim=1)
        fused_belief = (weights * beliefs).sum(dim=1, keepdim=True)
        log_u = torch.cat([torch.log(o['uncertainty'].clamp(min=1e-8)) for o in opinions], dim=1)
        fused_uncertainty = torch.exp(log_u.sum(dim=1, keepdim=True))
        return fused_belief.clamp(0, 1), fused_uncertainty, opinions

# ============================================================
# Evidential Loss
# ============================================================

def evidential_loss(evidence, target, epoch_ratio=1.0, kl_weight=0.1):
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    y = torch.cat([target, 1 - target], dim=1)
    p = alpha / S
    err = (y - p) ** 2
    var = p * (1 - p) / (S + 1)
    mse_loss = (err + var).sum(dim=1).mean()
    alpha_tilde = y + (1 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl_div = (torch.lgamma(S_tilde.squeeze(1))
              - torch.lgamma(alpha_tilde).sum(dim=1)
              + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde)
                 - torch.digamma(S_tilde))).sum(dim=1))
    return mse_loss + min(1.0, epoch_ratio) * kl_weight * kl_div.mean()

def evidential_loss_weighted(evidence, target, pixel_weight,
                              epoch_ratio=1.0, kl_weight=0.05):
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    y = torch.cat([target, 1 - target], dim=1)
    p = alpha / S
    err = (y - p) ** 2
    var = p * (1 - p) / (S + 1)
    per_pixel = (err + var).sum(dim=1, keepdim=True)
    weighted_loss = (per_pixel * pixel_weight).mean()
    alpha_tilde = y + (1 - y) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    kl_div = (torch.lgamma(S_tilde)
              - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
              + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde)
                 - torch.digamma(S_tilde))).sum(dim=1, keepdim=True))
    return weighted_loss + min(1.0, epoch_ratio) * kl_weight * (kl_div * pixel_weight).mean()

def belief_fusion_loss(fused_prob, fused_uncertainty, majority_vote, individual_targets):
    consensus = F.binary_cross_entropy(
        fused_prob.clamp(1e-6, 1 - 1e-6), majority_vote, reduction='mean')
    var = individual_targets.float().var(dim=1, keepdim=True)
    calib = F.mse_loss(fused_uncertainty.clamp(0, 1), torch.sigmoid(var * 10 - 1.25))
    return consensus + 0.5 * calib

def evidence_to_logit(evidence):
    alpha = evidence + 1
    return torch.log(alpha[:, 0:1].clamp(min=1e-6)) - torch.log(alpha[:, 1:2].clamp(min=1e-6))

def evidence_to_prob(evidence):
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    return alpha[:, 0:1] / S

def evidence_to_uncertainty(evidence):
    alpha = evidence + 1
    return 2.0 / alpha.sum(dim=1, keepdim=True)

def compute_disagreement_weight(masks, base_weight=1.0, boost_factor=5.0):
    var = masks.float().var(dim=1, keepdim=True)
    return base_weight + boost_factor * (var / 0.25).clamp(0, 1)

# ============================================================
# SABR
# ============================================================

class SpatialBoundaryDetector(nn.Module):
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
        ks = int(6 * sigma + 1) | 1
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        g2d = g1d[:, None] * g1d[None, :]
        g2d = g2d / g2d.sum()
        self.register_buffer('gauss_kernel', g2d.view(1, 1, ks, ks))
        self.ks_pad = ks // 2

    def _smooth(self, x):
        return F.conv2d(x, self.gauss_kernel, padding=self.ks_pad)

    @torch.no_grad()
    def forward(self, all_evidences):
        uncertainties, probs = [], []
        for e in all_evidences:
            alpha = e.detach() + 1
            S = alpha.sum(dim=1, keepdim=True)
            uncertainties.append(2.0 / S)
            probs.append(alpha[:, 0:1] / S)
        mean_u = torch.stack(uncertainties, dim=0).mean(dim=0)
        smooth_u = self._smooth(mean_u)
        disagreement = torch.stack(probs, dim=0).var(dim=0, unbiased=False)
        smooth_d = self._smooth(disagreement)
        def _norm(x):
            f = x.flatten(1)
            lo = f.min(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            hi = f.max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            return (x - lo) / (hi - lo + 1e-8)
        boundary_map = (0.5 * _norm(smooth_u) + 0.5 * _norm(smooth_d)).clamp(0, 1)
        buc = self._compute_buc(mean_u, boundary_map)
        return boundary_map, buc

    @staticmethod
    def _compute_buc(uncertainty_map, boundary_map, threshold=0.3):
        br = (boundary_map > threshold).float()
        nb = 1.0 - br
        mu_R = (uncertainty_map * br).sum() / br.sum().clamp(min=1)
        mu_nR = (uncertainty_map * nb).sum() / nb.sum().clamp(min=1)
        return (mu_R / (mu_R + mu_nR + 1e-8)).item()

class BoundaryRefiner(nn.Module):
    def __init__(self, feat_ch=16, style_dim=32, hidden_ch=16):
        super().__init__()
        self.input_proj = nn.Conv2d(feat_ch + 2, hidden_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, hidden_ch)
        self.style_film = nn.Sequential(
            nn.Linear(style_dim, hidden_ch), nn.GELU(),
            nn.Linear(hidden_ch, hidden_ch * 2))
        nn.init.zeros_(self.style_film[-1].weight)
        nn.init.zeros_(self.style_film[-1].bias)
        self.conv1 = nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, hidden_ch)
        self.output_proj = nn.Conv2d(hidden_ch, 2, 1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, seg_feat, boundary_map, evi_prob, style):
        x = torch.cat([seg_feat, boundary_map, evi_prob], dim=1)
        h = F.gelu(self.norm1(self.input_proj(x)))
        film = self.style_film(style)
        s, sh = film.chunk(2, dim=1)
        h = h * (1.0 + s[:, :, None, None]) + sh[:, :, None, None]
        h = F.gelu(self.norm2(self.conv1(h)))
        return torch.tanh(self.output_proj(h)) * 2.0 * boundary_map

def spatial_boundary_loss(refined_evidence, target, boundary_map):
    alpha = refined_evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha[:, 0:1] / S
    uncertainty = 2.0 / S
    pred_err = (prob - target).abs()
    calib_gap = (uncertainty - pred_err).abs()
    bdry_w = 1.0 + 4.0 * boundary_map
    weighted_calib = (calib_gap * bdry_w).mean()
    smooth_u = F.avg_pool2d(uncertainty, 5, stride=1, padding=2)
    bin_err = ((prob > 0.5).float() != target).float()
    smooth_e = F.avg_pool2d(bin_err, 5, stride=1, padding=2)
    space_loss = ((smooth_u - smooth_e).abs() * boundary_map).sum() / (boundary_map.sum() + 1e-8)
    bb = F.max_pool2d(boundary_map, 5, stride=1, padding=2).clamp(0, 1)
    pb, tb = prob * bb, target * bb
    inter = (pb * tb).sum(dim=(2, 3))
    union = pb.sum(dim=(2, 3)) + tb.sum(dim=(2, 3))
    dice = (1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)).mean()
    return weighted_calib + 0.5 * space_loss + 0.5 * dice

# ============================================================
# SC-ECRD
# ============================================================

class LatentDiffusionPrior(nn.Module):
    def __init__(self, z_dim=6, hidden_dim=128, style_dim=32):
        super().__init__()
        inp = z_dim * 2 + 1 + style_dim
        self.input_proj = nn.Linear(inp, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim); self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim); self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim); self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim); self.ln5 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.fc_out.weight); nn.init.zeros_(self.fc_out.bias)

    def forward(self, noisy_z, mu_prior, t_norm, style):
        x = torch.cat([noisy_z, mu_prior, t_norm.unsqueeze(1), style], dim=1)
        h = F.gelu(self.ln1(self.input_proj(x)))
        h = h + F.gelu(self.ln2(self.fc2(h)))
        h = h + F.gelu(self.ln3(self.fc3(h)))
        h = h + F.gelu(self.ln4(self.fc4(h)))
        h = h + F.gelu(self.ln5(self.fc5(h)))
        return self.fc_out(h)

class EarlyStoppingTracker:
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience = patience; self.min_delta = min_delta
        self.best_score = 0.0; self.best_epoch = 0; self.counter = 0

    def update(self, score, epoch):
        if score > self.best_score + self.min_delta:
            self.best_score = score; self.best_epoch = epoch; self.counter = 0
        else:
            self.counter += 1
        remaining = self.patience - self.counter
        return (self.counter == 0, remaining <= 0, remaining,
                f"  EarlyStopping: best={self.best_score:.4f} @ep{self.best_epoch}"
                f"  patience={self.counter}/{self.patience}  remaining={remaining}")

# ============================================================
# ============================================================

class MOSAIC(nn.Module):
    def __init__(self, input_channels=1, num_classes=1,
                 num_filters=[16, 32, 64, 128, 256],
                 latent_dim=6, no_convs_fcomb=4,
                 num_experts=4, reg_factor=1.0, original_backbone=True,
                 num_timesteps=200, inference_steps=1, diff_loss_weight=0.5,
                 style_bank_dim=32,
                 sparse_threshold_low=0.1, sparse_threshold_high=0.9, sparse_dilate=7,
                 prediction_type='x0', inference_t0_ratio=0.1,
                 boundary_refine_ch=16,
                 ecrd_train_steps=30, ecrd_inference_steps=5, ecrd_hidden_ch=64,
                 sabr_sigma=2.0,
                 ecrd_start_threshold=0.9, ecrd_full_threshold=0.3,
                 bdry_warmup_epochs=40, bdry_start_loss=0.8, bdry_full_loss=0.3,
                 ecrd_eta=0.8, ecrd_t0_ratio=0.6,
                 ecrd_diversity_weight=1.0,
                 pred_coverage_weight=0.3,
                 expert_recon_weight=0.05,
                 early_stopping_patience=30,
                 ecrd_fallback_epoch=40, ecrd_fallback_min_mix=0.15,
                 ecrd_activate_confirm_epochs=3,
                 ecrd_recon_activate_threshold=0.3,
                 ecrd_recon_t0_ratio=0.2,
                 ecrd_guidance_strength=0.5,
                 ecrd_target_noise_scale=0.3,
                 num_annotators=None, **kwargs):
        super(MOSAIC, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.num_experts = num_experts
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.reg_factor = reg_factor
        self.original_backbone = original_backbone
        if num_annotators is None:
            num_annotators = num_experts
        self.num_annotators = num_annotators
        self.diff_loss_weight = diff_loss_weight
        self.style_bank_dim = style_bank_dim

        self.ecrd_guidance_strength = ecrd_guidance_strength
        self.ecrd_target_noise_scale = ecrd_target_noise_scale

        if self.original_backbone:
            self.unet = Unet(self.input_channels, self.num_classes,
                             self.num_filters, self.initializers,
                             apply_last_layer=False, padding=True).to(device)
        else:
            self.unet = UnetHeadless(input_channels).to(device)

        self.prior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters,
            self.no_convs_per_block, self.latent_dim, self.initializers,
            num_annotators=self.num_annotators).to(device)
        self.posterior = AxisAlignedConvGaussian(
            self.input_channels, self.num_filters,
            self.no_convs_per_block, self.latent_dim,
            self.initializers, posterior=True,
            num_annotators=self.num_annotators).to(device)
        self.fcomb = EvidentialFcomb(
            self.num_filters, self.latent_dim,
            self.input_channels, self.num_classes,
            self.no_convs_fcomb,
            {'w': 'orthogonal', 'b': 'normal'},
            style_dim=style_bank_dim, use_tile=True).to(device)
        self.proj_heads = Projection(
            self.num_experts, self.latent_dim, style_dim=style_bank_dim).to(device)
        self.style_bank = ExpertStyleBank(
            num_experts=num_experts, style_dim=style_bank_dim).to(device)
        self.ecrd_style_proj = ECRDStyleProjector(style_dim=style_bank_dim).to(device)
        self.wbf = DifferentiableWBF()

        self.spatial_boundary_detector = SpatialBoundaryDetector(sigma=sabr_sigma).to(device)
        self.boundary_refiner = BoundaryRefiner(
            feat_ch=self.num_filters[0], style_dim=style_bank_dim,
            hidden_ch=boundary_refine_ch).to(device)
        self.use_boundary = True
        self.bdry_warmup_epochs = bdry_warmup_epochs
        self.bdry_start_loss = bdry_start_loss
        self.bdry_full_loss = bdry_full_loss
        self._bdry_mix_ratio = 0.0
        self._bdry_ready = False
        self._current_epoch = 0

        self.register_buffer('_saved_bdry_mix', torch.tensor(0.0))
        self.register_buffer('_saved_ecrd_mix', torch.tensor(0.0))

        self.ecrd_train_steps = ecrd_train_steps
        self.ecrd_inference_steps = ecrd_inference_steps
        self.diff_z_prior = LatentDiffusionPrior(
            z_dim=self.latent_dim, hidden_dim=ecrd_hidden_ch * 4,
            style_dim=style_bank_dim).to(device)
        self._rebuild_alpha_bars(ecrd_train_steps)
        self.ecrd_t0_ratio = ecrd_t0_ratio
        self.ecrd_recon_t0_ratio = ecrd_recon_t0_ratio
        self.ecrd_eta = ecrd_eta
        self.ecrd_diversity_weight = ecrd_diversity_weight
        self.pred_coverage_weight = pred_coverage_weight
        self.expert_recon_weight = expert_recon_weight
        self.use_ecrd = True
        self.ecrd_start_threshold = ecrd_start_threshold
        self.ecrd_full_threshold = ecrd_full_threshold
        self._ecrd_mix_ratio = 0.0
        self._ecrd_ready = False
        self.ecrd_fallback_epoch = ecrd_fallback_epoch
        self.ecrd_fallback_min_mix = ecrd_fallback_min_mix
        self.ecrd_activate_confirm_epochs = ecrd_activate_confirm_epochs
        self._ecrd_confirm_counter = 0
        self.ecrd_recon_activate_threshold = ecrd_recon_activate_threshold

        self.early_stopping = EarlyStoppingTracker(patience=early_stopping_patience)

        self.use_diffusion = False
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        self.inference_t0_ratio = inference_t0_ratio
        self.inference_steps = inference_steps
        self.noise_loss_val = 0.0
        self.style_loss_val = 0.0
        self.bound_loss_val = 0.0
        self.fusion_loss_val = 0.0
        self.bdry_loss_val = 0.0
        self.ecrd_loss_val = 0.0
        self.ecrd_div_val = 0.0
        self.ecrd_recon_val = 0.0
        self.pred_coverage_val = 0.0
        self.buc_val = 0.0
        self._epoch_ratio = 0.0
        self.ealr_noise_ratio = 0.0
        self.ealr_lambda = 0.0
        self.ecrd_grad_norm = 0.0

    def _rebuild_alpha_bars(self, num_steps):
        s = 0.008
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = (f[1:] / f[0]).float().clamp(min=1e-5, max=0.9999)
        if hasattr(self, 'ecrd_alpha_bars'):
            del self.ecrd_alpha_bars
        self.register_buffer('ecrd_alpha_bars', alpha_bars)
        self.ecrd_train_steps = num_steps

    def get_base_params(self):
        """Return the parameters of the base network (frozen in the auxiliary stage)."""
        return (
            list(self.unet.parameters())
            + list(self.prior.parameters())
            + list(self.posterior.parameters())
            + list(self.fcomb.conv1.parameters())
            + list(self.fcomb.mid_layers.parameters())
            + list(self.fcomb.last_layer.parameters())
        )

    def get_aux_params(self):
        """Return the parameters of the three auxiliary modules.
        """
        ebf = (
            list(self.fcomb.film_proj1.parameters())
            + list(self.fcomb.film_proj2.parameters())
            + list(self.fcomb.film2_style_proj.parameters())
            + list(self.fcomb.film2_proj.parameters())
            + list(self.fcomb.se.parameters())
            + list(self.style_bank.parameters())
            + list(self.fcomb.evi_last_layer.parameters())
        )
        ecrd = list(self.diff_z_prior.parameters()) + list(self.ecrd_style_proj.parameters())
        sabr = list(self.boundary_refiner.parameters())
        return {'ebf': ebf, 'ecrd': ecrd, 'sabr': sabr}

    def freeze_base(self):
        """Freeze the base network before the auxiliary training stage."""
        for p in self.get_base_params():
            p.requires_grad = False
        self.unet.eval()
        self.prior.eval()
        self.posterior.eval()

    def freeze_aux(self):
        """Freeze the auxiliary modules during the base training stage."""
        groups = self.get_aux_params()
        for grp in groups.values():
            for p in grp:
                p.requires_grad = False

    def safe_load_state_dict(self, state_dict, ecrd_train_steps=None):
        filtered = {k: v for k, v in state_dict.items() if 'ecrd_alpha_bars' not in k}
        if 'fcomb.last_layer.weight' in filtered:
            old_w = filtered['fcomb.last_layer.weight']
            if old_w.shape[0] == 2 and self.fcomb.last_layer.weight.shape[0] == 1:
                print(f"  [safe_load] Migrating last_layer: [2,*,*,*] -> [1,*,*,*] (taking channel 0)")
                filtered['fcomb.last_layer.weight'] = old_w[0:1]
                if 'fcomb.last_layer.bias' in filtered:
                    filtered['fcomb.last_layer.bias'] = filtered['fcomb.last_layer.bias'][0:1]

        result = self.load_state_dict(filtered, strict=False)
        self._rebuild_alpha_bars(ecrd_train_steps or self.ecrd_train_steps)

        if '_saved_bdry_mix' in state_dict:
            self._bdry_mix_ratio = state_dict['_saved_bdry_mix'].item()
            self._bdry_ready = self._bdry_mix_ratio > 0.01
            print(f"  [safe_load] Restored SABR mix_ratio={self._bdry_mix_ratio:.3f}")
        if '_saved_ecrd_mix' in state_dict:
            self._ecrd_mix_ratio = state_dict['_saved_ecrd_mix'].item()
            self._ecrd_ready = self._ecrd_mix_ratio > 0.01
            print(f"  [safe_load] Restored SC-ECRD mix_ratio={self._ecrd_mix_ratio:.3f}")

        real_missing = [k for k in result.missing_keys
                        if 'ecrd_alpha_bars' not in k
                        and '_saved_bdry_mix' not in k
                        and '_saved_ecrd_mix' not in k]
        if real_missing:
            print(f"  [safe_load] Missing keys: {real_missing}")
        if result.unexpected_keys:
            print(f"  [safe_load] Unexpected keys: {result.unexpected_keys}")
        return result

    class _DummyBASS:
        threshold_low = 0.1; threshold_high = 0.9; dilate_kernel = 7
        def parameters(self): return iter([])
    class _DummyModule:
        def parameters(self): return iter([])

    def __getattr__(self, name):
        if name == 'mask_denoiser': return self._DummyModule()
        if name == 'bass': return self._DummyBASS()
        if name in ('scheduler', 'ensemble', 'flow_net', 'evidence_head'):
            return self._DummyModule()
        return super().__getattr__(name)

    def update_epoch_state(self, epoch, total_epochs, avg_ecrd_loss=None):
        self._current_epoch = epoch
        self._epoch_ratio = epoch / max(total_epochs - 1, 1)
        ecrd_loss = avg_ecrd_loss if avg_ecrd_loss is not None else self.ecrd_loss_val

        print(f"  [SC-ECRD] ecrd_loss_val={ecrd_loss:.8f}  threshold={self.ecrd_start_threshold}")
        if ecrd_loss < 1e-6 or ecrd_loss >= self.ecrd_start_threshold:
            new_ecrd_mix = 0.0
            self._ecrd_confirm_counter = 0
        elif ecrd_loss <= self.ecrd_full_threshold:
            new_ecrd_mix = 1.0
            self._ecrd_confirm_counter += 1
        else:
            new_ecrd_mix = (self.ecrd_start_threshold - ecrd_loss) / \
                           (self.ecrd_start_threshold - self.ecrd_full_threshold)
            new_ecrd_mix = max(0.0, min(1.0, new_ecrd_mix))
            self._ecrd_confirm_counter += 1

        if self._ecrd_confirm_counter < self.ecrd_activate_confirm_epochs:
            new_ecrd_mix = 0.0

        if epoch >= self.ecrd_fallback_epoch and new_ecrd_mix < self.ecrd_fallback_min_mix:
            if ecrd_loss > 0 and ecrd_loss < self.ecrd_start_threshold:
                new_ecrd_mix = self.ecrd_fallback_min_mix

        if self._ecrd_mix_ratio < 0.01 and new_ecrd_mix > 0.01:
            print(f"  [SC-ECRD] ACTIVATING epoch {epoch} (loss={ecrd_loss:.4f} mix={new_ecrd_mix:.3f})")
        elif self._ecrd_mix_ratio > 0.01 and new_ecrd_mix < 0.01:
            print(f"  [SC-ECRD] DEACTIVATING epoch {epoch} (loss={ecrd_loss:.4f})")
        self._ecrd_mix_ratio = min(new_ecrd_mix, 1.0)
        self._ecrd_ready = self._ecrd_mix_ratio > 0.01

        bdry_loss = self.bdry_loss_val
        if epoch < self.bdry_warmup_epochs:
            new_bdry = 0.0
        elif bdry_loss >= self.bdry_start_loss:
            new_bdry = 0.0
        elif bdry_loss <= self.bdry_full_loss:
            new_bdry = 1.0
        else:
            new_bdry = (self.bdry_start_loss - bdry_loss) / \
                       (self.bdry_start_loss - self.bdry_full_loss)
            new_bdry = max(0.0, min(1.0, new_bdry))
        if self._bdry_mix_ratio == 0 and new_bdry > 0:
            print(f"  [SABR] ACTIVATING epoch {epoch} (loss={bdry_loss:.4f})")
        self._bdry_mix_ratio = 0.7 * self._bdry_mix_ratio + 0.3 * new_bdry
        self._bdry_ready = self._bdry_mix_ratio > 0.01

        self._saved_bdry_mix.fill_(self._bdry_mix_ratio)
        self._saved_ecrd_mix.fill_(self._ecrd_mix_ratio)

    def update_early_stopping(self, score, epoch):
        return self.early_stopping.update(score, epoch)

    # ============================================================
    # ============================================================

    def forward(self, patch, segm=None, training=True):
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        if self.original_backbone:
            self.unet_features = self.unet.forward(patch, False)
        else:
            self.unet_features = self.unet.forward(patch)

    def minmax(self, post_masks):
        return torch.cat([
            torch.max(post_masks, dim=1, keepdim=True).values,
            torch.min(post_masks, dim=1, keepdim=True).values], dim=1)

    def get_z_set(self, dist, sample_num=400):
        return torch.cat([dist.sample().unsqueeze(1) for _ in range(sample_num)], dim=1)

    def z_mapping(self, z_set):
        cases = []
        for idx in range(z_set.size(1)):
            cases.append(self.fcomb.forward(
                self.unet_features, z_set[:, idx],
                expert_style=None, use_softmax=False))
        samples = torch.cat(cases, dim=1)
        return torch.median(samples, dim=1, keepdim=True).values

    def z_mapping_evidential(self, z_set, expert_style):
        evi_sum = None
        for idx in range(z_set.size(1)):
            evi = self.fcomb.forward(self.unet_features, z_set[:, idx],
                                     expert_style=expert_style, use_softmax=False)
            evi_sum = evi if evi_sum is None else evi_sum + evi
        return evidence_to_logit(evi_sum / z_set.size(1))

    def z_mapping_evidential_with_evidence(self, z_set, expert_style):
        evi_sum = None
        for idx in range(z_set.size(1)):
            evi = self.fcomb.forward(self.unet_features, z_set[:, idx],
                                     expert_style=expert_style, use_softmax=False)
            evi_sum = evi if evi_sum is None else evi_sum + evi
        avg = evi_sum / z_set.size(1)
        return evidence_to_logit(avg), avg

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        if analytic:
            return kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        if calculate_posterior:
            z_posterior = self.posterior_latent_space.rsample()
        return (self.posterior_latent_space.log_prob(z_posterior)
                - self.prior_latent_space.log_prob(z_posterior))

    def prior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            z = self.prior_latent_space.rsample() if training else self.prior_latent_space.sample()
            samples.append(self.fcomb.forward(
                self.unet_features, z, expert_style=None, use_softmax=False))
        return samples

    def posterior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            z = self.posterior_latent_space.rsample() if training else self.posterior_latent_space.sample()
            samples.append(self.fcomb.forward(
                self.unet_features, z, expert_style=None, use_softmax=False))
        return samples

    def elbo(self, args, segm, criterion, analytic_kl=True):
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl))
        self.re_post = self.posterior_sampling(sample_num=1, training=True)[0]
        self.re_prior = torch.cat(
            self.prior_sampling(sample_num=args.prior_sample_num, training=True), dim=1)
        prior_bound = self.minmax(segm)
        prior_pred_max = torch.max(self.re_prior, dim=1, keepdim=True).values
        prior_pred_min = torch.min(self.re_prior, dim=1, keepdim=True).values
        pred_bound = torch.cat([prior_pred_max, prior_pred_min], dim=1)
        bound_loss = criterion(pred_bound, prior_bound)
        random_label = segm[:, np.random.randint(0, args.mask_num)].unsqueeze(1)
        self.reconstruction_loss = torch.sum(criterion(self.re_post, random_label))
        self.bound_loss = torch.sum(bound_loss)
        self.bound_loss_val = self.bound_loss.item()
        return -(self.reconstruction_loss + self.kl + args.beta * self.bound_loss)

    def combined_loss(self, args, labels, loss_fct):
        elbo = self.elbo(args, labels, criterion=loss_fct)
        self.reg_loss = (
            l2_regularisation(self.posterior) + l2_regularisation(self.prior)
            + l2_regularisation(self.fcomb.conv1) + l2_regularisation(self.fcomb.mid_layers)
        ) * self.reg_factor
        return -elbo + self.reg_loss

    # ============================================================
    # SC-ECRD
    # ============================================================

    def _get_ecrd_style(self, expert_idx_tensor):
        return self.ecrd_style_proj(self.style_bank(expert_idx_tensor))

    def _compute_expert_target_z(self, mu_prior, mu_post, sigma_prior, k, B, dev):
        style_k = self._get_ecrd_style(
            torch.full((B,), k, dtype=torch.long, device=dev)).detach()
        style_dir = F.normalize(style_k, dim=1)

        z_dim = mu_prior.shape[1]
        n_groups = math.ceil(style_dir.shape[1] / z_dim)
        pad_len = n_groups * z_dim - style_dir.shape[1]
        style_padded = F.pad(style_dir, (0, pad_len))
        dir_proj = style_padded.view(B, z_dim, n_groups).mean(dim=2)
        dir_proj = F.normalize(dir_proj, dim=1)

        expert_shift = self.ecrd_guidance_strength * sigma_prior.detach() * dir_proj
        noise = self.ecrd_target_noise_scale * sigma_prior.detach() * torch.randn_like(mu_prior)

        target_z = mu_prior + expert_shift + noise
        return target_z.detach()

    def _diffusion_prior_loss(self, z_target, mu_prior_d, style_d):
        B = z_target.shape[0]
        dev = z_target.device
        t_max = max(2, int(self.ecrd_t0_ratio * (self.ecrd_train_steps - 1)) + 2)
        t = torch.randint(0, t_max, (B,), device=dev)
        t_norm = t.float() / max(self.ecrd_train_steps - 1, 1)
        ab = self.ecrd_alpha_bars[t].unsqueeze(1)
        noise = torch.randn_like(z_target)
        noisy_z = torch.sqrt(ab) * z_target + torch.sqrt(1.0 - ab) * noise
        pred_z0 = self.diff_z_prior(noisy_z, mu_prior_d, t_norm, style_d)
        return F.mse_loss(pred_z0, z_target)

    def _style_diversity_reg(self, mu_prior_d, denoising_loss_val):
        B = mu_prior_d.shape[0]; dev = mu_prior_d.device
        if denoising_loss_val > 0.5:
            return torch.tensor(0.0, device=dev)
        scale = max(0.0, min(1.0, (0.5 - denoising_loss_val) / 0.3))
        margin = 0.5
        total = torch.tensor(0.0, device=dev)
        for _ in range(3):
            t_idx = torch.randint(1, self.ecrd_train_steps, (1,)).item()
            ab_t = self.ecrd_alpha_bars[t_idx]
            noise = torch.randn_like(mu_prior_d)
            z_noisy = (torch.sqrt(ab_t) * mu_prior_d + torch.sqrt(1.0 - ab_t) * noise).detach()
            t_norm = torch.full((B,), t_idx / max(self.ecrd_train_steps - 1, 1), device=dev)
            preds = []
            for k in range(self.num_experts):
                sk = self._get_ecrd_style(
                    torch.full((B,), k, dtype=torch.long, device=dev)).detach()
                preds.append(self.diff_z_prior(z_noisy, mu_prior_d, t_norm, sk))
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    total = total + F.relu(margin - (preds[i] - preds[j]).pow(2).mean())
        n_pairs = self.num_experts * (self.num_experts - 1) // 2
        return total / (3 * n_pairs) * scale

    def _diffusion_prior_loss_multisample(self, mu_prior, masks, n_samples=2):
        B = mu_prior.shape[0]; dev = mu_prior.device
        mu_d = mu_prior.detach()
        mu_post = self.posterior_latent_space.mean.detach()
        sigma_prior = self.prior_latent_space.stddev.detach()

        total_dn = 0.0; count = 0
        for k in range(self.num_experts):
            sk = self._get_ecrd_style(
                torch.full((B,), k, dtype=torch.long, device=dev)).detach()
            for _ in range(n_samples):
                z_target_k = self._compute_expert_target_z(
                    mu_d, mu_post, sigma_prior, k, B, dev)
                total_dn += self._diffusion_prior_loss(z_target_k, mu_d, sk)
                count += 1
        dn_loss = total_dn / count

        with torch.no_grad():
            guidance_var = (self.ecrd_guidance_strength ** 2) * sigma_prior.pow(2).mean()
            noise_var = (self.ecrd_target_noise_scale ** 2) * sigma_prior.pow(2).mean()
            z_var = (guidance_var + noise_var).clamp(min=1e-4)
        dn_val_normalized = (dn_loss.detach() / z_var).clamp(max=2.0).item()
        self.ecrd_loss_val = dn_val_normalized

        div_reg = self._style_diversity_reg(mu_d, dn_val_normalized)
        self.ecrd_div_val = div_reg.item()

        recon = torch.tensor(0.0, device=dev)
        if dn_val_normalized < self.ecrd_recon_activate_threshold:
            recon = self._expert_reconstruction_loss(mu_d, masks)
        self.ecrd_recon_val = recon.item()

        return dn_loss + self.ecrd_diversity_weight * div_reg + self.expert_recon_weight * recon

    def _sample_z_diffusion_with_grad(self, mu_prior_d, style_d):
        B = mu_prior_d.shape[0]; dev = mu_prior_d.device
        t0 = max(1, min(int(self.ecrd_recon_t0_ratio * (self.ecrd_train_steps - 1)),
                        self.ecrd_train_steps - 1))
        ab = self.ecrd_alpha_bars[t0]
        noise = torch.randn_like(mu_prior_d)
        z_noisy = torch.sqrt(ab) * mu_prior_d + torch.sqrt(1.0 - ab) * noise
        t_norm = torch.full((B,), t0 / max(self.ecrd_train_steps - 1, 1), device=dev)
        return self.diff_z_prior(z_noisy, mu_prior_d, t_norm, style_d)

    def _expert_reconstruction_loss(self, mu_prior_d, masks):
        B = mu_prior_d.shape[0]; dev = mu_prior_d.device
        unet_d = self.unet_features.detach()
        num_mc = masks.shape[1]
        total = 0.0
        for k in range(self.num_experts):
            ecrd_sk = self._get_ecrd_style(
                torch.full((B,), k, dtype=torch.long, device=dev)).detach()
            z_k = self._sample_z_diffusion_with_grad(mu_prior_d, ecrd_sk)
            ebf_sk = self.style_bank(
                torch.full((B,), k, dtype=torch.long, device=dev)).detach()
            evi_k = self.fcomb.forward(unet_d, z_k, expert_style=ebf_sk, use_softmax=False,
                                       detach_base=True)
            total += evidential_loss(evi_k, masks[:, k % num_mc: k % num_mc + 1].float(),
                                     self._epoch_ratio, kl_weight=0.05)
        return total / self.num_experts

    @torch.no_grad()
    def _sample_z_diffusion(self, mu_prior, style):
        B = mu_prior.shape[0]; dev = mu_prior.device
        t0 = max(1, min(int(self.ecrd_t0_ratio * (self.ecrd_train_steps - 1)),
                        self.ecrd_train_steps - 1))
        ab_t0 = self.ecrd_alpha_bars[t0]
        z = torch.sqrt(ab_t0) * mu_prior + torch.sqrt(1.0 - ab_t0) * torch.randn_like(mu_prior)
        steps = np.clip(np.linspace(t0, 0, self.ecrd_inference_steps + 1).astype(int),
                        0, self.ecrd_train_steps - 1)
        for i in range(len(steps) - 1):
            tc, tn = steps[i], steps[i + 1]
            if tc == tn: continue
            t_norm = torch.full((B,), tc / max(self.ecrd_train_steps - 1, 1), device=dev)
            ab_c = self.ecrd_alpha_bars[tc]
            pred_z0 = self.diff_z_prior(z, mu_prior, t_norm, style)
            if tn > 0:
                ab_n = self.ecrd_alpha_bars[tn]
                eps = (z - torch.sqrt(ab_c) * pred_z0) / torch.sqrt(1.0 - ab_c).clamp(min=1e-6)
                sig = self.ecrd_eta * torch.sqrt((1 - ab_n) / (1 - ab_c)) * \
                      torch.sqrt((1 - ab_c / ab_n).clamp(min=0))
                dc = (1.0 - ab_n - sig ** 2).clamp(min=0)
                z = torch.sqrt(ab_n) * pred_z0 + torch.sqrt(dc) * eps + sig * torch.randn_like(z)
            else:
                z = pred_z0
        return z

    @torch.no_grad()
    def _sample_z_mixed(self, mu_prior, ecrd_style):
        z_g = self.prior_latent_space.sample()
        m = self._ecrd_mix_ratio
        if m < 0.01:
            return z_g
        z_d = self._sample_z_diffusion(mu_prior, ecrd_style)
        expert_offset = z_d - mu_prior
        return z_g + m * expert_offset

    def _apply_boundary_refinement(self, evidence, boundary_map, style):
        evi_prob = evidence_to_prob(evidence)
        delta = self.boundary_refiner(
            self.unet_features.detach(), boundary_map, evi_prob, style.detach())
        return (evidence + self._bdry_mix_ratio * delta).clamp(min=0.01)

    def _prediction_coverage_loss(self, all_evidences, masks, unet_feat_detached):
        pred_probs = torch.stack(
            [evidence_to_prob(e.detach()) for e in all_evidences], dim=1)
        pred_var = pred_probs.var(dim=1)
        gt_var = masks.float().var(dim=1, keepdim=True)
        coverage_mse = F.mse_loss(pred_var, gt_var)

        align_loss = torch.tensor(0.0, device=masks.device)
        disagree_mask = (gt_var > 0.01).float().detach()
        n_disagree = disagree_mask.sum().clamp(min=1.0)
        num_mc = masks.shape[1]
        for idx in range(len(all_evidences)):
            pred_p = evidence_to_prob(all_evidences[idx])
            target = masks[:, idx % num_mc: idx % num_mc + 1].float()
            align_loss = align_loss + ((pred_p - target).abs() * disagree_mask).sum() / n_disagree
        align_loss = align_loss / len(all_evidences)

        return coverage_mse.detach() + 0.3 * align_loss

    def _evidential_auxiliary_loss(self, args, masks):
        """Auxiliary-stage loss. The base network is frozen, so this loss only updates the auxiliary modules."""
        B = masks.shape[0]; dev = masks.device
        num_experts = args.mask_num

        unet_feat_detached = self.unet_features.detach()

        prior_ratio = min(0.3, self._epoch_ratio * 0.5)
        if torch.rand(1).item() < prior_ratio:
            z = self.prior_latent_space.rsample().detach()
        else:
            z = self.posterior_latent_space.rsample().detach()

        pixel_weight = compute_disagreement_weight(masks)
        majority_vote = (masks.float().mean(dim=1, keepdim=True) >= 0.5).float()

        all_evidences = []
        for idx in range(num_experts):
            expert_style = self.style_bank(
                torch.full((B,), idx, dtype=torch.long, device=dev))
            all_evidences.append(self.fcomb.forward(
                unet_feat_detached, z, expert_style=expert_style, use_softmax=False,
                detach_base=True))

        ealr_warmup = 0.3; ealr_max = 0.8
        if self._epoch_ratio <= ealr_warmup:
            noise_lambda = 0.0
        else:
            noise_lambda = ealr_max * (self._epoch_ratio - ealr_warmup) / (1.0 - ealr_warmup)

        num_mc = masks.shape[1]
        total_evi = 0.0; total_nr = 0.0
        for idx in range(num_experts):
            target = masks[:, idx % num_mc: idx % num_mc + 1].float()
            evidence = all_evidences[idx]
            with torch.no_grad():
                alpha = evidence.detach() + 1
                S = alpha.sum(dim=1, keepdim=True)
                confidence = (1.0 - 2.0 / S).clamp(0, 1)
                noise_score = (target != majority_vote).float() * (1.0 - confidence)
                lqw = (1.0 - noise_lambda * noise_score).clamp(min=0.2)
                total_nr += (lqw < 0.8).float().mean().item()
            total_evi += evidential_loss_weighted(
                evidence, target, pixel_weight * lqw, self._epoch_ratio, kl_weight=0.05)
        total_evi /= num_experts
        self.noise_loss_val = total_evi.item()
        self.ealr_noise_ratio = total_nr / num_experts
        self.ealr_lambda = noise_lambda

        fused_prob, fused_unc, _ = self.wbf(all_evidences)
        fusion_loss = belief_fusion_loss(fused_prob, fused_unc, majority_vote, masks)
        self.fusion_loss_val = fusion_loss.item()

        div_loss = torch.tensor(0.0, device=dev); pc = 0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                mi, mj = i % num_mc, j % num_mc
                td = (masks[:, mi:mi+1].float() - masks[:, mj:mj+1].float()).abs().mean()
                ed = (evidence_to_prob(all_evidences[i])
                      - evidence_to_prob(all_evidences[j])).abs().mean()
                div_loss = div_loss + F.relu(td - ed); pc += 1
        div_loss = div_loss / max(pc, 1)
        ortho = self.style_bank.orthogonality_loss()

        coverage = self._prediction_coverage_loss(all_evidences, masks, unet_feat_detached)
        self.pred_coverage_val = coverage.item()

        style_loss = (0.4 * div_loss + 0.01 * ortho + 0.3 * fusion_loss
                      + self.pred_coverage_weight * coverage)
        self.style_loss_val = style_loss.item()

        mu_prior = self.prior_latent_space.mean
        ecrd_loss = self._diffusion_prior_loss_multisample(mu_prior, masks, n_samples=2)

        boundary_map, buc = self.spatial_boundary_detector(all_evidences)
        self.buc_val = buc
        bdry_loss = torch.tensor(0.0, device=dev)
        for idx in range(num_experts):
            es = self.style_bank(
                torch.full((B,), idx, dtype=torch.long, device=dev)).detach()
            ep = evidence_to_prob(all_evidences[idx].detach())
            delta = self.boundary_refiner(
                unet_feat_detached, boundary_map.detach(), ep, es)
            refined = (all_evidences[idx].detach() + delta).clamp(min=0.01)
            bdry_loss += spatial_boundary_loss(
                refined, masks[:, idx % num_mc: idx % num_mc + 1].float(), boundary_map.detach())
        bdry_loss /= num_experts
        self.bdry_loss_val = bdry_loss.item()

        return 0.5 * total_evi + 0.3 * style_loss + 0.1 * ecrd_loss + 0.1 * bdry_loss

    # ============================================================
    # ============================================================

    def train_step(self, args, images, masks, loss_fct, stage=1):
        if stage == 1 or stage == '1a':
            self.forward(images, masks, training=True)
            original_loss = self.combined_loss(args, masks, loss_fct)
            self.noise_loss_val = self.style_loss_val = self.fusion_loss_val = 0.0
            self.bdry_loss_val = self.ecrd_loss_val = self.ecrd_div_val = 0.0
            self.ecrd_recon_val = self.pred_coverage_val = self.buc_val = 0.0
            self.ealr_noise_ratio = self.ealr_lambda = 0.0
            return original_loss, self.re_post

        elif stage == '1b':
            with torch.no_grad():
                self.forward(images, masks, training=True)
                _ = self.combined_loss(args, masks, loss_fct)
            evi_loss = self._evidential_auxiliary_loss(args, masks)
            return evi_loss, self.re_post

        elif stage == 2 or stage == '2':
            self.forward(images, None, training=False)
            z_set = self.get_z_set(self.prior_latent_space, sample_num=100)
            B = images.shape[0]; dev = images.device
            num_mc = masks.shape[1]
            samples_fc = []
            for idx in range(self.num_experts):
                z_exp = self.proj_heads(self.unet_features, z_set, idx)
                samples_fc.append(self.z_mapping(z_exp))
            dice_fc = loss_fct(torch.cat(samples_fc, dim=1), masks)

            samples_evi = []; all_avg = []; evi_ft = 0.0
            for idx in range(self.num_experts):
                es = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
                ze = self.proj_heads(self.unet_features, z_set, idx, expert_style=es)
                se, ae = self.z_mapping_evidential_with_evidence(ze, es)
                samples_evi.append(se); all_avg.append(ae)
                evi_ft += evidential_loss(ae, masks[:, idx % num_mc: idx % num_mc + 1].float(),
                                          1.0, kl_weight=0.05)
            evi_ft /= self.num_experts
            dice_evi = loss_fct(torch.cat(samples_evi, dim=1), masks)

            bmap, buc = self.spatial_boundary_detector(all_avg); self.buc_val = buc
            bdry = torch.tensor(0.0, device=dev)
            for idx in range(self.num_experts):
                es = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev)).detach()
                ep = evidence_to_prob(all_avg[idx].detach())
                delta = self.boundary_refiner(self.unet_features.detach(), bmap.detach(), ep, es)
                ref = (all_avg[idx].detach() + delta).clamp(min=0.01)
                bdry += spatial_boundary_loss(
                    ref, masks[:, idx % num_mc: idx % num_mc + 1].float(), bmap.detach())
            bdry /= self.num_experts
            total = dice_fc + dice_evi + 0.3 * evi_ft + 0.15 * bdry
            self.noise_loss_val = evi_ft.item()
            self.style_loss_val = self.fusion_loss_val = 0.0
            self.bdry_loss_val = bdry.item()
            self.ecrd_loss_val = self.ecrd_div_val = self.ecrd_recon_val = 0.0
            return total, torch.cat(samples_fc, dim=1)

    # ============================================================
    # ============================================================

    @torch.no_grad()
    def _inference_evidential(self, sample_num):
        B, dev = self.unet_features.shape[0], self.unet_features.device
        mu_prior = self.prior_latent_space.mean
        K = self.num_experts

        ebf_styles = []
        ecrd_styles = []
        for kk in range(K):
            idx_t = torch.full((B,), kk, dtype=torch.long, device=dev)
            ebf_styles.append(self.style_bank(idx_t))
            ecrd_styles.append(self._get_ecrd_style(idx_t))

        all_individual = []
        per_expert_evi_sums = {kk: None for kk in range(K)}
        per_expert_counts = {kk: 0 for kk in range(K)}

        for i in range(sample_num):
            eidx = i % K
            if self.use_ecrd and self._ecrd_mix_ratio > 0.01:
                z = self._sample_z_mixed(mu_prior, ecrd_styles[eidx])
            else:
                z = self.prior_latent_space.sample()
            evidence = self.fcomb.forward(
                self.unet_features, z,
                expert_style=ebf_styles[eidx], use_softmax=False)
            all_individual.append((evidence, eidx))
            if per_expert_evi_sums[eidx] is None:
                per_expert_evi_sums[eidx] = evidence
            else:
                per_expert_evi_sums[eidx] = per_expert_evi_sums[eidx] + evidence
            per_expert_counts[eidx] += 1

        if self.use_boundary and self._bdry_mix_ratio > 0.01:
            avg_list = []
            for kk in range(K):
                if per_expert_counts[kk] > 0:
                    avg_list.append(per_expert_evi_sums[kk] / per_expert_counts[kk])
                else:
                    z = self.prior_latent_space.sample()
                    avg_list.append(self.fcomb.forward(
                        self.unet_features, z,
                        expert_style=ebf_styles[kk], use_softmax=False))
            boundary_map, _ = self.spatial_boundary_detector(avg_list)
            results = []
            for evidence, eidx in all_individual:
                refined = self._apply_boundary_refinement(
                    evidence, boundary_map, ebf_styles[eidx])
                logit = evidence_to_logit(refined)
                T = getattr(self, 'inference_temperature', 1.0)
                if T != 1.0:
                    logit = logit / T
                results.append(logit)
        else:
            T = getattr(self, 'inference_temperature', 1.0)
            results = []
            for e, _ in all_individual:
                logit = evidence_to_logit(e)
                if T != 1.0:
                    logit = logit / T
                results.append(logit)

        return torch.cat(results, dim=1)

    @torch.no_grad()
    def _inference_hybrid(self, sample_num):
        """
        """
        B, dev = self.unet_features.shape[0], self.unet_features.device
        mu_prior = self.prior_latent_space.mean
        K = self.num_experts

        ebf_styles = []
        ecrd_styles = []
        for kk in range(K):
            idx_t = torch.full((B,), kk, dtype=torch.long, device=dev)
            ebf_styles.append(self.style_bank(idx_t))
            ecrd_styles.append(self._get_ecrd_style(idx_t))

        results = []
        for i in range(sample_num):
            eidx = i % K
            if self.use_ecrd and self._ecrd_mix_ratio > 0.01:
                z = self._sample_z_mixed(mu_prior, ecrd_styles[eidx])
            else:
                z = self.prior_latent_space.sample()
            logit = self.fcomb.forward(
                self.unet_features, z,
                expert_style=None, use_softmax=False)
            results.append(logit)

        return torch.cat(results, dim=1)

    @torch.no_grad()
    def val_step(self, images, sample_num=50):
        self.forward(images, None, training=False)
        if not self.use_diffusion:
            return torch.cat(
                self.prior_sampling(sample_num=sample_num, training=False), dim=1)
        return self._inference_evidential(sample_num)

    def test_step(self, images, sample_num=100):
        """
        """
        force_ecrd = getattr(self, 'force_ecrd_inference', False)
        if self.use_diffusion and force_ecrd:
            self.forward(images, None, training=False)
            if getattr(self, 'inference_use_evidence_head', False):
                return self._inference_evidential(sample_num)
            sabr_blend = getattr(self, 'inference_sabr_blend', False)
            return self._inference_hybrid(sample_num, sabr_blend=sabr_blend)
        self.forward(images, None, training=False)
        z_set = self.get_z_set(self.prior_latent_space, sample_num=sample_num)
        samples = []
        for idx in range(self.num_experts):
            z_exp = self.proj_heads(self.unet_features, z_set, idx)
            samples.append(self.z_mapping(z_exp))
        return torch.cat(samples, dim=1)

    @torch.no_grad()
    def predict_with_uncertainty(self, images, sample_num=100):
        self.forward(images, None, training=False)
        z_set = self.get_z_set(self.prior_latent_space, sample_num=sample_num)
        B, dev = images.shape[0], images.device
        expert_evs = []
        for idx in range(self.num_experts):
            es = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
            ze = self.proj_heads(self.unet_features, z_set, idx, expert_style=es)
            evi_sum = None
            for j in range(ze.size(1)):
                evi = self.fcomb.forward(self.unet_features, ze[:, j],
                                         expert_style=es, use_softmax=False)
                evi_sum = evi if evi_sum is None else evi_sum + evi
            expert_evs.append(evi_sum / ze.size(1))
        if self.use_boundary and self._bdry_mix_ratio > 0.01:
            bm, _ = self.spatial_boundary_detector(expert_evs)
            for idx in range(self.num_experts):
                s = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
                expert_evs[idx] = self._apply_boundary_refinement(expert_evs[idx], bm, s)
        probs, uncs = [], []
        for e in expert_evs:
            a = e + 1; S = a.sum(dim=1, keepdim=True)
            probs.append(a[:, 0:1] / S); uncs.append(2.0 / S)
        fp, fu, _ = self.wbf(expert_evs)
        return torch.cat(probs, dim=1), torch.cat(uncs, dim=1), fp, fu

    # ============================================================
    # ============================================================

    @torch.no_grad()
    def get_visualization_data(self, images, masks=None, sample_num=20):
        """
        """
        self.forward(images, None, training=False)
        B, dev = images.shape[0], images.device
        mu_prior = self.prior_latent_space.mean
        K = self.num_experts

        all_evidences = []
        per_expert_samples_prob = []
        for idx in range(K):
            style = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
            ecrd_s = self._get_ecrd_style(
                torch.full((B,), idx, dtype=torch.long, device=dev))
            evi_sum = None
            sp = []
            for _ in range(sample_num):
                if self.use_ecrd and self._ecrd_mix_ratio > 0.01:
                    z = self._sample_z_mixed(mu_prior, ecrd_s)
                else:
                    z = self.prior_latent_space.sample()
                evi = self.fcomb.forward(self.unet_features, z,
                                         expert_style=style, use_softmax=False)
                evi_sum = evi if evi_sum is None else evi_sum + evi
                sp.append(evidence_to_prob(evi))
            all_evidences.append(evi_sum / sample_num)
            per_expert_samples_prob.append(torch.stack(sp, dim=1))  # [B, sample_num, 1, H, W]

        expert_probs = [evidence_to_prob(e) for e in all_evidences]
        expert_uncs = [evidence_to_uncertainty(e) for e in all_evidences]

        saved_ecrd = self._ecrd_mix_ratio
        self._ecrd_mix_ratio = 0.0
        gauss_evidences = []
        for idx in range(K):
            style = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
            evi_sum = None
            for _ in range(sample_num):
                z = self.prior_latent_space.sample()
                evi = self.fcomb.forward(self.unet_features, z,
                                         expert_style=style, use_softmax=False)
                evi_sum = evi if evi_sum is None else evi_sum + evi
            gauss_evidences.append(evi_sum / sample_num)
        self._ecrd_mix_ratio = saved_ecrd
        gauss_probs = [evidence_to_prob(e) for e in gauss_evidences]

        bmap, buc = self.spatial_boundary_detector(all_evidences)
        refined_evidences = []
        if self._bdry_mix_ratio > 0.01:
            for idx in range(K):
                style = self.style_bank(torch.full((B,), idx, dtype=torch.long, device=dev))
                refined_evidences.append(self._apply_boundary_refinement(
                    all_evidences[idx], bmap, style))
        else:
            refined_evidences = list(all_evidences)
        refined_probs = [evidence_to_prob(e) for e in refined_evidences]

        fp, fu, _ = self.wbf(all_evidences)

        stacked = torch.stack(expert_probs, dim=1)  # [B, K, 1, H, W]
        pred_var = stacked.squeeze(2).var(dim=1, keepdim=True)
        gt_var = None
        if masks is not None:
            gt_var = masks.float().var(dim=1, keepdim=True)

        return {
            'input': images,
            'gt_masks': masks.float() if masks is not None else None,
            'expert_probs': torch.stack(expert_probs, dim=1).squeeze(2),  # [B,K,H,W]
            'expert_uncertainties': torch.stack(expert_uncs, dim=1).squeeze(2),
            'expert_probs_refined': torch.stack(refined_probs, dim=1).squeeze(2),
            'gauss_expert_probs': torch.stack(gauss_probs, dim=1).squeeze(2),  # [B,K,H,W]
            'per_expert_samples_prob': per_expert_samples_prob,  # list of [B, sample_num, 1, H, W]
            'fused_prob': fp,
            'fused_uncertainty': fu,
            'boundary_map': bmap,
            'pred_variance_map': pred_var,
            'gt_variance_map': gt_var,
            'buc': buc,
            'ecrd_mix': self._ecrd_mix_ratio,
            'bdry_mix': self._bdry_mix_ratio,
        }

    @torch.no_grad()
    def get_diversity_visualization_data(self, images, masks=None, sample_num=20):
        return self.get_visualization_data(images, masks, sample_num)

    @torch.no_grad()
    def get_expert_samples_data(self, images, n_samples=8):
        return self.get_visualization_data(images, None, n_samples)