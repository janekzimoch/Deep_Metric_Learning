import matplotlib.pyplot as plt
import numpy as np
import sys
import json

stats = json.load(open('./performance/VGG16__model_0_V6_SEEN_V2', 'r'))
stats2 = json.load(open('./performance/VGG16__model_0_V6_UNSEEN', 'r'))
plt.rcParams.update({'font.size': 14.5})

names = ['VGG16__model_0_V6_SEEN_V2', 'VGG16__model_0_V6_UNSEEN']
exp_name = 'VGG'

def plot_metric_vs_metric(metric_1='recall', metric_2='false_rate', xlim=(0,1), ylim=(0,1)):
    for model_name, data, color, label in zip(names, [stats, stats2], ['tab:blue', 'tab:green'], ['seen categories', 'unseen categories']):
        x1 = data[model_name][metric_1]
        x2 = data[model_name][metric_2]
        plt.plot(x1, x2, color=color, label=label)

    plt.legend()
    plt.xlabel(metric_1)
    plt.ylabel(metric_2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig('../figures/final/precision_recall_{}_final.png'.format(exp_name))

plot_metric_vs_metric(metric_1='recall', metric_2='precision', xlim=(0.1,1), ylim=(0.5,1))
