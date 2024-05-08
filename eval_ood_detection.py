import os
import argparse
import numpy as np
import torch
from scipy import stats
import config
from utils.common import setup_seed, get_and_print_results, print_measures
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.dataloaders_utils import set_few_shot_loader, set_val_loader, set_ood_loader_ImageNet
from utils.id_like import get_prompts, get_result, load_model


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates OOD for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root-dir', default="your datasets path", type=str,
                        help='root dir of datasets')
    parser.add_argument('--in_dataset', default='ImageNet', type=str, help='in-distribution dataset')
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--score', default='id-like', type=str)
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                        help='which pretrained img encoder to use')
    parser.add_argument('--n_shot', default=1, type=int,
                        help="how many samples are used to estimate classwise mean and precision matrix")
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--test_batch_size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--n_crop', default=256, type=int, help='crop num')
    parser.add_argument('--n_selection', default=32, type=int, help='selection num')
    # parser.add_argument('--selection_p', default=0.2, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ex_prompts', default=100, type=int, help='number of extra prompts')
    parser.add_argument('--n_epoch', default=3, type=int, help='number of epoch')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--lam_in', default=1.0, type=float, help='lambda of id loss')
    parser.add_argument('--lam_out', default=0.3, type=float, help='lambda of ood loss')
    parser.add_argument('--lam_diff', default=0.2, type=float, help='lambda of difference')

    args = parser.parse_args()

    args.n_cls = config.data_info[args.in_dataset]['n_cls']

    args.log_directory = f"results/{args.in_dataset}/id-like"

    os.makedirs(args.log_directory, exist_ok=True)
    setup_seed(args.seed)
    return args


def train():
    args = process_args()

    log = setup_log(args)
    out_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']

    test_labels = config.data_info[args.in_dataset]['labels']
    ex_labels = ['X'] * args.n_ex_prompts

    model_checkpoint_save_path = os.path.join(args.log_directory, 'model_checkpoint.pth')

    if os.path.exists(model_checkpoint_save_path):
        model = load_model(args, test_labels, ex_labels)
    else:
        few_shot_loader = set_few_shot_loader(args)
        model = get_prompts(args, few_shot_loader, test_labels, ex_labels)

    test_loader = set_val_loader(args)
    result_in = get_result(args, model, test_loader, test_labels, ex_labels, if_acc=True)
    score_in = result_in['scores']
    acc = result_in['acc']
    log.debug(f"Acc: {acc}")

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset)
        result_out = get_result(args, model, ood_loader, test_labels, ex_labels)
        score_out = result_out['scores']
        log.debug(f"in scores: {stats.describe(score_in)}")
        log.debug(f"out scores: {stats.describe(score_out)}")
        plot_distribution(args, score_in, score_out, out_dataset)
        get_and_print_results(args, log, score_in, score_out,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == "__main__":
    train()
