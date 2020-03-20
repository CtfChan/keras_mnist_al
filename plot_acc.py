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

acquisition_iterations = 99
num_of_queries = 10


plt.axis([0, 1000, 0.1, 1])
plt.yticks(np.array(range(11))*0.02 + 0.8)
plt.xticks(np.array(range(10))*100)
for a in af_color.keys():
    acc_list = []
    for i in range(1, 2):
        file_str = './results/' + a + '_' + str(i) + '_test_acc.npy'
        acc = np.load(file_str)
        acc_list.append(acc)
    acc_mean = np.mean(acc_list, axis=0)
    plt.plot(np.array(range(acquisition_iterations))*num_of_queries, acc_mean, label=a, marker='x')

plt.ylabel('Test set accuracy')
plt.xlabel('Number of additional data points used')
plt.grid()
plt.title('Comparison of Various Acquisition Functions on MNIST using Reversed Metric')
plt.legend(loc=0)
plt.savefig("mnist_al_results")
plt.show()
