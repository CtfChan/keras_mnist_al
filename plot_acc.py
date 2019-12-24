import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# v1 = np.load('./results/VAR_RATIO_1_test_acc.npy')
# v2 = np.load('./results/VAR_RATIO_2_test_acc.npy')
# v3 = np.load('./results/VAR_RATIO_3_test_acc.npy')
# v4 = np.load('./results/VAR_RATIO_4_test_acc.npy')
# v5 = np.load('./results/VAR_RATIO_5_test_acc.npy')

# r1 = np.load('./results/RANDOM_1_test_acc.npy')
# r2 = np.load('./results/RANDOM_2_test_acc.npy')
# r3 = np.load('./results/RANDOM_3_test_acc.npy')
# r4 = np.load('./results/RANDOM_4_test_acc.npy')
# r5 = np.load('./results/RANDOM_5_test_acc.npy')

# e1 = np.load('./results/ENTROPY_1_test_acc.npy')
# e2 = np.load('./results/ENTROPY_2_test_acc.npy')
# e3 = np.load('./results/ENTROPY_3_test_acc.npy')
# e4 = np.load('./results/ENTROPY_4_test_acc.npy')
# e5 = np.load('./results/ENTROPY_5_test_acc.npy')

# v = np.mean([v1, v2, v3, v4, v5], axis=0)
# r = np.mean([r1, r2, r3, r4, r5], axis=0)
# e = np.mean([e1, e2, e3, e4, e5], axis=0)

# plt.axis([0, 1000, 0.8, 1])
# plt.yticks(np.array(range(11))*0.02 + 0.8)
# plt.xticks(np.array(range(10))*100)
# plt.plot(np.array(range(99))*10, v, label='var-ratio')
# plt.plot(np.array(range(99))*10, r, 'g', label='random')
# plt.plot(np.array(range(99))*10, e, 'r', label='entropy')
# plt.ylabel('Test set accuracy')
# plt.xlabel('Number of data points used')
# plt.grid()
# plt.title('Comparison of Various Acquisition Function')
# plt.legend(loc=0)
# plt.show()

#TODO add margin sampling when available
af = ['ENTROPY', 'RANDOM', 'VAR_RATIO', 'BALD', 'MEAN_STD']

# af_color = {'ENTROPY': 'r',
#             'RANDOM': 'g',
#             'VAR_RATIO': 'b',
#             'BALD': 'o',
#             'MEAN_STD': 'p',
#             'MARGIN_SAMPLING': 'y'}
af_color = {'ENTROPY': 'r',
            'RANDOM': 'g',
            'VAR_RATIO': 'b',
            'BALD': 'o',
            'MEAN_STD': 'p'}

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
    plt.plot(np.array(range(99))*10, acc_mean, label=a)

plt.ylabel('Test set accuracy')
plt.xlabel('Number of data points used')
plt.grid()
plt.title('Comparison of Various Acquisition Function')
plt.legend(loc=0)
plt.show()