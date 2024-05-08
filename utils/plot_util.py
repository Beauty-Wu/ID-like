
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import seaborn as sns
import torch.nn.functional as F

def plot_distribution(args, id_scores, ood_scores, out_dataset):
    # sns.set(style="white", palette="muted")
    # palette = ['#A8BAE3', '#55AB83']
    # sns.displot({"ID":-1 * id_scores, "OOD": -1 * ood_scores}, label="id", kind = "kde", palette=palette, fill = True, alpha = 0.8)
    # plt.savefig(os.path.join(args.log_directory,f"{args.score}_{out_dataset}.png"), bbox_inches='tight')

    # sns.kdeplot(-1 * ood_scores, label="OOD", color='#0070C0', fill=True, alpha=0.5)
    # sns.kdeplot(-1 * id_scores, label="ID", color='#55AB83', fill=True, alpha=0.5)
    # plt.legend()
    # plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.png"))
    # plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.svg"), format='svg')
    # plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.pdf"), format='svg')
    # plt.close()
    sns.kdeplot(-1 * ood_scores, color='#0070C0', fill=True, alpha=0.5, cut=0, clip=(0., 1.))
    sns.kdeplot(-1 * id_scores, color='#55AB83', fill=True, alpha=0.5, cut=0, clip=(0., 1.))
    plt.ticklabel_format(axis='both', style="sci", scilimits=(0, 0))
    plt.tick_params(labelsize=16)
    plt.ylabel('')
    plt.savefig(os.path.join(args.log_directory, f"{args.score}_{out_dataset}.png"))
    plt.close()


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize=9) 
    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


