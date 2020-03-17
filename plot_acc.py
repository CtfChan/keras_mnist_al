import matplotlib
from matplotlib import pyplot as plt
import numpy as np

af_color = {'ENTROPY': 'r',
            'RANDOM': 'g',
            'VAR_RATIO': 'b',
            'BALD': 'o',
            'MEAN_STD': 'p',
            'MARGIN_SAMPLING': 'c',
            'CLASSIFICATION_STABILITY': 'm'}

plt.axis([0, 1000, 0.8, 1])
plt.yticks(np.array(range(11))*0.02 + 0.8)
plt.xticks(np.array(range(10))*100)
for a in af_color.keys():
    acc_list = []
    for i in range(1, 6):
        file_str = './results/' + a + '_' + str(i) + '_test_acc.npy'
        acc = np.load(file_str)
        acc_list.append(acc)
    acc_mean = np.mean(acc_list, axis=0)
    plt.plot(np.array(range(99))*10, acc_mean, label=a, marker='x')

plt.ylabel('Test set accuracy')
plt.xlabel('Number of data points used')
plt.grid()
plt.title('Comparison of Various Acquisition Function on MNIST')
plt.legend(loc=0)
plt.savefig("mnist_al_results")
plt.show()
