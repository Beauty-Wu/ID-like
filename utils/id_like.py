import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.id_like_utils import ClipPromptLearner
from utils.id_like_loss import get_loss
from utils.common import AverageMeter, accuracy
from utils import imagenet_templates
from clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import config


def select_in_out(args, image_features, sim):

    idx_in = torch.topk(sim, dim=0, k=args.n_selection)[1].squeeze()
    image_features_crop_in_temp = torch.index_select(image_features, index=idx_in, dim=0)
    idx_out = torch.topk(-sim, dim=0, k=args.n_selection)[1].squeeze()
    image_features_crop_out_temp = torch.index_select(image_features, index=idx_out, dim=0)
    image_features_in_temp = image_features_crop_in_temp
    image_features_out_temp = image_features_crop_out_temp

    return image_features_in_temp, image_features_out_temp


def get_in_out(args, clip, model, labels, images, targets):

    image_features_in = []
    image_features_out = []
    targets_in = []
    targets_out = []
    with torch.no_grad():

        for image_idx, (image, target) in enumerate(zip(images, targets)):
            label = labels[target.item()]
            # openai_imagenet_template = imagenet_templates.openai_imagenet_template
            openai_imagenet_template = [lambda c: f'a photo of a {c}.']
            select_prompts_in = [func(label) for func in openai_imagenet_template]
            text_inputs = tokenize(select_prompts_in).cuda()
            select_prompts_in = clip.encode_text(text_inputs)
            select_prompts_in /= select_prompts_in.norm(dim=-1, keepdim=True)

            image = image.cuda()
            target = target.cuda()
            image_features = model.get_image_features(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            sim = image_features @ select_prompts_in.t()
            sim = torch.max(sim, dim=1, keepdim=True)[0]

            image_features_in_temp, image_features_out_temp = select_in_out(args, image_features, sim)
            image_features_in.append(image_features_in_temp)
            image_features_out.append(image_features_out_temp)

            # create in target
            targets_in_temp = torch.tile(target, dims=(image_features_in_temp.size(0),))
            targets_in.append(targets_in_temp)

            # create out target
            # no use
            prompt_features = model.get_text_features()
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
            prompt_features_out = prompt_features[args.n_cls:, ...]
            logit_out_temp = image_features_out_temp @ prompt_features_out.t()
            targets_out_temp = torch.max(logit_out_temp, dim=1)[1] + args.n_cls
            targets_out.append(targets_out_temp)

        image_features_in = torch.cat(image_features_in, dim=0)
        image_features_out = torch.cat(image_features_out, dim=0)
        targets_in = torch.cat(targets_in, dim=0).cuda()
        targets_out = torch.cat(targets_out, dim=0).cuda()
    return image_features_in, image_features_out, targets_in, targets_out


def get_prompts(args, loader, labels, ex_labels):
    model = ClipPromptLearner(args,
                              classnames=labels, ex_classnames=ex_labels, arch=args.CLIP_ckpt, device='cuda',
                              n_ctx=config.n_ctx, ctx_init=config.ctx_init,
                              ctx_position=config.ctx_position, learned_cls=config.learned_cls,
                              n_ex_ctx=config.n_ex_ctx, ex_ctx_init=config.ex_ctx_init,
                              ex_ctx_position=config.ex_ctx_position, ex_learned_cls=config.ex_learned_cls)

    loss_meter = AverageMeter()
    optimizer = torch.optim.AdamW([{'params': model.prompt_learner.parameters()},
                                   {'params': model.ex_prompt_learner.parameters()}], args.lr)

    clip, _, _ = load(args.CLIP_ckpt, device='cuda', download_root=config.DOWNLOAD_ROOT)

    for epoch in range(args.n_epoch):

        tqdm.write(f'Train epoch:{epoch + 1}/{args.n_epoch}')
        for batch_idx, (images, targets) in enumerate(tqdm(loader)):
            image_features_in, image_features_out, targets_in, targets_out = \
                get_in_out(args, clip, model, labels, images, targets)

            # train
            # get prompts
            logit_scale = model.logit_scale.exp()
            prompt_features = model.get_text_features()
            prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)
            loss, loss_str = get_loss(args, prompt_features,
                                      image_features_in, image_features_out,
                                      targets_in, targets_out, logit_scale)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_meter.update(loss.detach().cpu().item())
            tqdm.write(f'Train epoch:{epoch + 1}/{args.n_epoch}\t'
                       f'Loss_avg:{loss_meter.avg:.6f}\t' + loss_str)

        if epoch+1 == args.n_epoch:
            model_save_dir = args.log_directory
            os.makedirs(model_save_dir, exist_ok=True)
            model_checkpoint_save_path = os.path.join(model_save_dir, 'model_checkpoint.pth')
            model_checkpoint = {
                'prompt_learner_state_dict': model.prompt_learner.state_dict(),
                'ex_prompt_learner_state_dict': model.ex_prompt_learner.state_dict(),
            }
            torch.save(model_checkpoint, model_checkpoint_save_path)

    return model


def get_result(args, model, loader, labels, ex_labels, if_acc=False):
    tqdm_object = tqdm(loader, total=len(loader))
    outputs = []
    all_targets = []
    result = {
        'scores': None,
        'acc': None,
    }

    with torch.no_grad():
        text_features = model.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for batch_idx, (images, targets) in enumerate(tqdm_object):
        with torch.no_grad():
            images = images.cuda()
            targets = targets.long().cuda()
            image_features = model.image_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            output = logit_scale * image_features @ text_features.t()

            output = output.detach().cpu()
            outputs.append(output)
            all_targets.append(targets)
    outputs = torch.cat(outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # scores

    outputs_softmax = F.softmax(outputs, dim=1)
    scores = torch.sum(outputs_softmax[:, args.n_cls:], dim=1).detach().cpu().squeeze().numpy() - 1

    result['scores'] = scores
    # acc
    if if_acc:
        res = accuracy(outputs[:, :args.n_cls], all_targets.detach().cpu())
        result['acc'] = [acc.item() for acc in res]
    return result


def load_model(args, labels, ex_labels):
    model = ClipPromptLearner(args,
                              classnames=labels, ex_classnames=ex_labels, arch=args.CLIP_ckpt, device='cuda',
                              n_ctx=config.n_ctx, ctx_init=config.ctx_init,
                              ctx_position=config.ctx_position, learned_cls=config.learned_cls,
                              n_ex_ctx=config.n_ex_ctx, ex_ctx_init=config.ex_ctx_init,
                              ex_ctx_position=config.ex_ctx_position, ex_learned_cls=config.ex_learned_cls)
    model_checkpoint_save_path = os.path.join(args.log_directory, 'model_checkpoint.pth')
    model_checkpoint = torch.load(model_checkpoint_save_path)
    model.prompt_learner.load_state_dict(model_checkpoint['prompt_learner_state_dict'])
    model.ex_prompt_learner.load_state_dict(model_checkpoint['ex_prompt_learner_state_dict'])
    return model.cuda()

