import json
import matplotlib.pyplot as plt


# exp_name = 'all_models'
# # file_names = ['4096-4096', '2048-4096', '2048-2048', '1024-2048', '1024-1024']
# file_names = ['0', '1', '2', '3', '4', '5']
# # names = ['m. #1', 'm. #2', 'm. #3', 'm. #4', 'm. #5', 'm. #6']
# file_names = ['ResNet_50__model_', 'ResNet_18__model_', 'VGG16__model_0']
# names = ['ResNet-50', 'ResNet-18', 'VGG-16']
# file_names = ['64-128-256-512-512', '32-64-128-256-512', '64-64-128-128-256']
file_names = ['VGG16__model_0_V4_test_20', 'VGG16__model_0_V4_test_40', 'VGG16__model_0_V4_test_60', 'VGG16__model_0_V4_test_80']
names = ['M=20', 'M=40', 'M=60', 'M=80']

exp_name = 'pair_mining'
plt.rcParams.update({'font.size': 16.5})



def plot_training(plot_type='val_accuracy'):
    plt.figure(figsize=(8,4))
    for f_name, name in zip(file_names, names):
        dir_name = '../model_history/exp_20_12/{}'.format(f_name)
        history = json.load(open(dir_name, 'r'))

        num_epochs = len(history[plot_type])
        plt.plot(range(num_epochs+1), [0.799] + history[plot_type], label=name)

    plt.ylim(0.4,0.85)
    plt.xlim(0,10)
    plt.ylabel('Validation Accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.tight_layout()



# # PLOT ACCURACY
# plt.figure()
# plot_training(plot_type='accuracy')
# plt.savefig('./figures/Architecture_exp/acc_{}.png'.format(file_name))

# PLOT LOSS
plt.figure()
plot_training(plot_type='val_accuracy1')
plt.savefig('../figures/Arch_exp_final/val_loss_{}.png'.format(exp_name))