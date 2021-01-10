import matplotlib.pyplot as plt
import numpy as np
import sys
import json

stats = json.load(open('./performance/VGG16__model_0_V6_SEEN_V2', 'r'))
stats2 = json.load(open('./performance/VGG16__model_0_V6_UNSEEN', 'r'))
plt.rcParams.update({'font.size': 14.5})
# names = ['m. #1', 'm. #2', 'm. #3', 'm. #4', 'm. #5', 'm. #6']
# names = ['ResNet-50', 'ResNet-18', 'VGG-16']
names = ['VGG16__model_0_V6_SEEN']
exp_name = 'VGG'

def plot_metric(stats, stats2, metric='accuracy'):
    plt.figure()
    for model_name, name in zip(['VGG16__model_0_V6_SEEN_V2'], ['seen categories']):
        thr = np.arange(0,1,0.01)
        acc = stats[model_name][metric]
        # print(acc)
        plt.plot(thr, acc, color='tab:blue', label=name)


    for model_name, name in zip(['VGG16__model_0_V6_UNSEEN'], ['unseen categories']):
        thr = np.arange(0,1,0.01)
        acc2 = stats2[model_name][metric]
        # print(acc2)
        plt.plot(thr, acc2, color='tab:green', label=name)
    # plt.legend()
    plt.ylabel(metric)
    plt.xlabel('normalized threshold')
    plt.grid(alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(0.5, 0.95)
    plt.tight_layout()
    plt.legend()
    plt.savefig('../figures/final/accuracy_{}_final.png'.format(exp_name))


plot_metric(stats, stats2, metric='accuracy')


def plot_distance_distribution(stats):
    distance = np.array(stats['VGG16__model_0_V6_SEEN_V2']['distance'])
    labels = np.array(stats['VGG16__model_0_V6_SEEN_V2']['labels'])
    
    print(distance)
    plt.figure()
    indx_positive = np.where(labels == 1)
    indx_negative = np.where(labels == 0)
    n = plt.hist(distance[indx_positive], bins=30, label='similar')
    m = plt.hist(distance[indx_negative], bins=30, label='dis-similar', alpha=0.5)
    
    mean_distance = np.mean(distance[indx_negative] - distance[indx_positive])
    print('Mean distance: ' + str(mean_distance))
    
    plt.legend()
    plt.ylim(0,max(int(np.max(n[0])), np.max(m[0]))+100)
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.tight_layout()
    # plt.title('Contrastive')
    # plt.xlim(0, 2.0)
    plt.savefig('../figures/final/dist_distribution_{}_final_SEEN.png'.format(exp_name))

plot_distance_distribution(stats)

def plot_distance_distribution(stats):
    distance = np.array(stats['VGG16__model_0_V6_UNSEEN']['distance'])
    labels = np.array(stats['VGG16__model_0_V6_UNSEEN']['labels'])
    
    print(distance)
    plt.figure()
    indx_positive = np.where(labels == 1)
    indx_negative = np.where(labels == 0)
    n = plt.hist(distance[indx_positive], bins=30, label='similar')
    m = plt.hist(distance[indx_negative], bins=30, label='dis-similar', alpha=0.5)
    
    mean_distance = np.mean(distance[indx_negative] - distance[indx_positive])
    print('Mean distance: ' + str(mean_distance))
    
    plt.legend()
    plt.ylim(0,max(int(np.max(n[0])), np.max(m[0]))+100)
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.tight_layout()

    # plt.title('Contrastive')
    # plt.xlim(0, 2.0)
    plt.savefig('../figures/final/dist_distribution_{}_final_UNSEEN.png'.format(exp_name))

plot_distance_distribution(stats2)
    
# def plot_metric_vs_metric(metric_1='recall', metric_2='false_rate', xlim=(0,1), ylim=(0,1)):
#     for model_name, color in zip(models_to_evaluate, ['b', 'r', 'g']):
#         x1 = models_to_evaluate[model_name][metric_1]
#         x2 = models_to_evaluate[model_name][metric_2]
#         plt.plot(x1, x2, color=color, label=model_name)
#         acc = models_to_evaluate[model_name]['accuracy']
#         plt.plot(x1, acc, '--', color=color)
#     plt.legend()
#     plt.xlabel(metric_1)
#     plt.ylabel(metric_2)
#     plt.xlim(xlim)
#     plt.ylim(ylim)
#     plt.show()