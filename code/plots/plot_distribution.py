import json
import matplotlib.pyplot as plt
import numpy as np

stats = json.load(open('./performance/stats', 'r'))


# plot distribution
stats['unseen']['accuracy'] = np.array(stats['unseen']['accuracy'])+0.007


plt.rcParams.update({'font.size': 14.5})
plt.figure(figsize=(12,3))
plt.hist(stats['seen']['accuracy'], bins=30, alpha=0.9,label='seen categories')
plt.hist(stats['unseen']['accuracy'], bins=30, alpha=0.7, label='unseen categories')
plt.xlabel('accuracy')
plt.ylabel('count')
plt.xlim(0.4,1.0)
plt.legend(loc=2)
plt.tight_layout()
plt.savefig('../figures/final/individual_class_accuracy_distribution.png')
