"""
initialize_optimization.py (Path A)
====================================

"""

import torch
from torch import nn

# ============================================================
# ============================================================

class _AbstractDiceLoss(nn.Module):
    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        raise NotImplementedError

    def forward(self, input, target):
        input = self.normalization(input)
        per_channel_dice = self.dice(input, target, weight=self.weight)
        return 1. - torch.mean(per_channel_dice)

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size()
        input = flatten(input)
        target = flatten(target)
        target = target.float()
        if input.size(0) == 1:
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False
        intersect = (input * target).sum(-1)
        intersect = intersect * w_l
        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        return 2 * (intersect / denominator)

# ============================================================
# ============================================================

def init_optimization(model, args):
    learning_rate = 0.0001

    if args.model_name == 'MOSAIC' and args.stage == 1:
        opt_params = [
            {'params': model.unet.parameters(), 'lr': learning_rate},
            {'params': model.prior.parameters(), 'lr': learning_rate},
            {'params': model.posterior.parameters(), 'lr': learning_rate},
            {'params': model.fcomb.parameters(), 'lr': learning_rate},
            {'params': model.mask_denoiser.parameters(), 'lr': learning_rate * 2},
        ]

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MOSAIC Stage I] Total: {total:,}, Trainable: {trainable:,}")
        print("  base components: unet + prior + posterior + fcomb")
        print(f"  auxiliary components: mask_denoiser (lr={learning_rate * 2})")

    elif args.model_name == 'MOSAIC' and args.stage == 2:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.proj_heads.parameters():
            param.requires_grad = True

        for param in model.mask_denoiser.parameters():
            param.requires_grad = True

        opt_params = [
            {'params': model.proj_heads.parameters(), 'lr': learning_rate},
            {'params': model.mask_denoiser.parameters(), 'lr': learning_rate * 0.5},
        ]

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[MOSAIC Stage II] Trainable: {trainable:,}, Total: {total:,}")

    elif args.model_name == 'pionono':
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.head.parameters()},
            {'params': model.z.parameters(), 'lr': 0.02}
        ]
    elif 'cm' in args.model_name:
        opt_params = [
            {'params': model.seg_model.parameters()},
            {'params': model.cm_head.parameters(), 'lr': 0.01}
        ]
    elif args.model_name == 'DPersona' and args.stage == 1:
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.prior.parameters()},
            {'params': model.posterior.parameters()},
            {'params': model.fcomb.parameters()}
        ]
    elif args.model_name == 'DPersona' and args.stage == 2:
        opt_params = [{'params': model.proj_heads.parameters()}]
    elif args.model_name == 'prob_unet':
        opt_params = [{'params': model.parameters()}]
    else:
        opt_params = [{'params': model.parameters()}]

    optimizer = torch.optim.Adam(opt_params, lr=learning_rate)
    loss_fct = GeneralizedDiceLoss(normalization='sigmoid')

    return optimizer, loss_fct