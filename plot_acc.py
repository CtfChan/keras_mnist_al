import matplotlib
from matplotlib import pyplot as plt
import numpy as np

v1 = np.load('var_ratio_test_acc.npy')
v2 = np.load('var_ratio_test_acc.npy')
v3 = np.load('var_ratio_test_acc.npy')
v4 = np.load('var_ratio_test_acc.npy')
v5 = np.load('var_ratio_test_acc.npy')

print(v1.shape)

r1 = np.load('random_test_acc.npy')
r2 = np.load('random_test_acc.npy')
r3 = np.load('random_test_acc.npy')
r4 = np.load('random_test_acc.npy')
r5 = np.load('random_test_acc.npy')

e1 = np.load('entropy_test_acc.npy')
e2 = np.load('entropy_test_acc.npy')
e3 = np.load('entropy_test_acc.npy')
e4 = np.load('entropy_test_acc.npy')
e5 = np.load('entropy_test_acc.npy')

v = np.mean([v1, v2, v3, v4, v5], axis=0)
r = np.mean([r1, r2, r3, r4, r5], axis=0)
e = np.mean([e1, e2, e3, e4, e5], axis=0)

print(e)
plt.axis([0, 1000, 0.8, 1])
plt.yticks(np.array(range(11))*0.02 + 0.8)
plt.xticks(np.array(range(10))*100)
plt.plot(np.array(range(99))*10, v, label='var-ratio')
plt.plot(np.array(range(99))*10, r, 'g', label='random')
plt.plot(np.array(range(99))*10, e, 'r', label='entropy')
plt.ylabel('Test set accuracy')
plt.xlabel('Number of data points used')
plt.grid()
plt.title('Comparison of Various Acquisition Function')
plt.legend(loc=0)
plt.show()