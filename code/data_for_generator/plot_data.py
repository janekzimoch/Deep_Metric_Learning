import matplotlib.pyplot as plt

# TEST WHETHER THE DATA SET PARTITION IS CORRECT
def plot_image(L, R, label):
    # i = 4
    print('label: ' + str(label))
    # print(train_left_label[i])
    plt.figure()
    plt.imshow(L)
    plt.savefig('./figures/{}_L.png'.format(str(label)))  

    plt.figure()  
    plt.imshow(R)
    plt.savefig('./figures/{}_R.png'.format(str(label)))    
    