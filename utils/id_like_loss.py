import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.id_like_utils import ClipPromptLearner
from utils.common import AverageMeter, accuracy
import config


def get_loss(args, prompt_features, image_features_in, image_features_out, targets_in, targets_out, logit_scale):
    prompt_features_in = prompt_features[:args.n_cls, ...]
    prompt_features_out = prompt_features[args.n_cls:, ...]
    # loss_in
    logit_in = logit_scale * image_features_in @ prompt_features.t()
    # logit_in = logit_scale * image_features_in @ prompt_features_in.t()
    loss_in = F.cross_entropy(logit_in, targets_in)

    # loss_out
    logit_out = logit_scale * image_features_out @ prompt_features.t()

    # logit_out_softmax_probs = F.softmax(logit_out, dim=1)
    # flag_out = torch.cat([torch.LongTensor([0] * args.n_cls + [1] * args.n_ex_prompts)], dim=0).cuda()
    # logit_out_softmax_probs_in = torch.sum(logit_out_softmax_probs * (1 - flag_out), dim=1)
    # logit_out_softmax_probs_in_log = -torch.log(1.-logit_out_softmax_probs_in)
    # loss_out = torch.mean(logit_out_softmax_probs_in_log)

    logit_out_softmax_probs = F.softmax(logit_out, dim=1)
    flag_out = torch.cat([torch.LongTensor([0] * args.n_cls + [1] * args.n_ex_prompts)], dim=0).cuda()
    logit_out_softmax_probs_in = torch.sum(logit_out_softmax_probs * (1 - flag_out), dim=1)
    logit_out_softmax_probs_in_log = torch.log(logit_out_softmax_probs_in + 1e-16)
    loss_out = torch.mean(logit_out_softmax_probs_in_log)

    # loss_diff
    loss_diff = torch.FloatTensor([0.]).cuda()
    for p in range(prompt_features_out.size(0) - 1):
        for q in range(p + 1, prompt_features_out.size(0)):
            loss_diff += F.cosine_embedding_loss(input1=prompt_features_out[p].unsqueeze(dim=0),
                                                 input2=prompt_features_out[q].unsqueeze(dim=0),
                                                 target=torch.LongTensor([-1]).cuda())
    if prompt_features_out.size(0) > 1:
        loss_diff /= (prompt_features_out.size(0) * (prompt_features_out.size(0) - 1) / 2.)

    # loss
    loss = loss_in * args.lam_in + loss_out * args.lam_out + loss_diff * args.lam_diff
    loss_str = f'Loss_now:{loss.detach().cpu().item():.6f}\t' \
               f'Loss_in:{loss_in.detach().cpu().item():.6f}\t' \
               f'Loss_out:{loss_out.detach().cpu().item():.6f}\t' \
               f'Loss_diff:{loss_diff.detach().cpu().item():.6f}'

    return loss, loss_str
