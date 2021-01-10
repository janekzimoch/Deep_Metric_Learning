import json
import matplotlib.pyplot as plt

# train_L = np.array()
# train_A = np.array()
# val_L = np.array()
# val_A = np.array()

file_names_1 = ['VGG16__model_0']

for f_name in file_names_1:
    dir_name = '../model_history/exp_20_12/{}'.format(f_name)
    history = json.load(open(dir_name, 'r'))
    train_L = history['loss']
    train_A = history['accuracy1']
    val_L = history['val_loss']
    val_A = history['val_accuracy1']

# epochs 21-40
t_L = [0.2195,0.2175,0.2158,0.2138,0.2123,0.2115,0.2105,0.2091,0.2084,0.2065,0.2038,0.2018,0.2015,0.1985,0.1985,0.1971,0.1965,0.1946,0.1944,0.1941]
t_A = [0.6504,0.6491,0.6575,0.6628,0.6650,0.6642,0.6669,0.6704,0.6712,0.6750,0.6832,0.6846,0.6877,0.6929,0.6938,0.6972,0.6968,0.7012,0.7011,0.7011]
v_L = [0.2162,0.2173,0.2178,0.2118,0.2106,0.2087,0.2091,0.2123,0.2012,0.2114,0.1967,0.2011,0.2054,0.2005,0.1989,0.1954,0.1968,0.1952,0.1891,0.1939]
v_A = [0.6478,0.6484,0.6492,0.6673,0.6671,0.6783,0.6629,0.6571,0.6907,0.6577,0.6945,0.6841,0.6715,0.6847,0.6947,0.6963,0.6955,0.6935,0.7161,0.6955]
train_L = list(train_L) + t_L
train_A = list(train_A) + t_A
val_L   = list(val_L) + v_L
val_A   = list(val_A) + v_A


# epochs 41-60
t_L = [0.1926,0.1917,0.1888,0.1897,0.1884,0.1867,0.1866,0.1846,0.1841,0.1815,0.1824,0.1823,0.1796,0.1794,0.1794,0.1791,0.1773,0.1743,0.1753,0.1720]
t_A = [0.7067,0.7070,0.7132,0.7107,0.7136,0.7171,0.7174,0.7216,0.7219,0.7257,0.7221,0.7255,0.7308,0.7316,0.7307,0.7311,0.7336,0.7396,0.7398,0.7459]
v_L = [0.1938,0.1934,0.1986,0.1855,0.1928,0.1909,0.1814,0.1872,0.1863,0.1853,0.1855,0.1830,0.1779,0.1841,0.1775,0.1800,0.1820,0.1860,0.1805,0.1751]
v_A = [0.7067,0.7077,0.6895,0.7194,0.7011,0.7051,0.7272,0.7133,0.7147,0.7143,0.7153,0.7254,0.7326,0.7125,0.7318,0.7318,0.7165,0.7212,0.7296,0.7356]
train_L = list(train_L) + t_L
train_A = list(train_A) + t_A
val_L   = list(val_L) + v_L
val_A   = list(val_A) + v_A

# epochs 61-80
t_L = [0.1731,0.1707,0.1707,0.1710,0.1698,0.1680,0.1684,0.1652,0.1633,0.1656,0.1614,0.1617,0.1602,0.1590,0.1583,0.1574,0.1564,0.1552,0.1537,0.1536]
t_A = [0.7424,0.7489,0.7463,0.7475,0.7518,0.7539,0.7539,0.7584,0.7614,0.7557,0.7661,0.7628,0.7682,0.7696,0.7714,0.7735,0.7741,0.7775,0.7781,0.7792]
v_L = [0.1733,0.1747,0.1694,0.1755,0.1706,0.1714,0.1776,0.1699,0.1706,0.1673,0.1667,0.1701,0.1700,0.1669,0.1658,0.1651,0.1636,0.1617,0.1595,0.1628]
v_A = [0.7450,0.7436,0.7552,0.7318,0.7480,0.7430,0.7416,0.7504,0.7412,0.7550,0.7534,0.7478,0.7410,0.7522,0.7516,0.7562,0.7568,0.7646,0.7746,0.7674]
train_L = list(train_L) + t_L
train_A = list(train_A) + t_A
val_L   = list(val_L) + v_L
val_A   = list(val_A) + v_A

file_names_2 = ['VGG16__model_0_V2', 'VGG16__model_0_V3', 'VGG16__model_0_V4', 'VGG16__model_0_V5', 'VGG16__model_0_V6','VGG16__model_0_V7','VGG16__model_0_V8']

for f_name in file_names_2:
    dir_name = '../model_history/exp_20_12/{}'.format(f_name)
    history = json.load(open(dir_name, 'r'))
    train_L = list(train_L) + list(history['loss'])
    train_A = list(train_A) + list(history['accuracy1'])
    val_L = list(val_L) + list(history['val_loss'])
    val_A = list(val_A) + list(history['val_accuracy1'])

train_L = train_L[:-101]
train_A = train_A[:-101]
val_L   = val_L[:-101]
val_A   = val_A[:-101]


def plot_training_loss():

    num_epochs = len(train_L)
    plt.plot(range(1,num_epochs+1), train_L, color='tab:green',label='train loss')
    plt.plot(range(1,num_epochs+1), val_L, color='tab:blue',label='validation loss')
    # plt.plot(range(1,num_epochs+1), val_A, '--',color='tab:blue',label='validation accuracy')

    # plt.ylim(0.24,0.32)
    # plt.xlim(1,20)
    # plt.ylabel('Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()

def plot_training_accuracy():

    num_epochs = len(train_L)
    plt.plot(range(1,num_epochs+1), train_A, color='tab:green',label='train accuracy')
    plt.plot(range(1,num_epochs+1), val_A, color='tab:blue',label='validation accuracy')

    # plt.ylim(0.24,0.32)
    # plt.xlim(1,20)
    # plt.ylabel('Validation loss')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(loc=4)
    plt.grid(alpha=0.5)
    plt.tight_layout()





# PLOT LOSS
plt.rcParams.update({'font.size': 14.5})

plt.figure(figsize=(12,4))
plot_training_loss()
plt.savefig('../figures/final/training_loss.png')

plt.figure(figsize=(12,4))
plot_training_accuracy()
plt.savefig('../figures/final/training_accuracy.png')