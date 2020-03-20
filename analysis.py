
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

for a in af_color.keys():
    print("Aquisition Function: ", a)
    acc_list = []
    for i in range(1, 2):
        file_str = './results/no_reverse/' + a + '_' + str(i) + '_test_acc.npy'
        acc = np.load(file_str)
        acc_list.append(acc)
    acc_mean = np.mean(acc_list, axis=0)
    
    
    acq_iter = np.linspace(1, acquisition_iterations, acquisition_iterations)

    interp_10p_error = np.interp(0.9, acc_mean, acq_iter)
    interp_5p_error = np.interp(0.95, acc_mean, acq_iter)

    print("10 percent error iterations: ", np.ceil(interp_10p_error)*10)
    print("5 percent error iterations : ", np.ceil(interp_5p_error)*10)

# 

    #     print("10 percent error: ", )

    # plt.plot(np.array(range(acquisition_iterations+1))*num_of_queries, acc_mean, label=a, marker='x')

    #     interp = np.interp(mAP, baseline_acc, num_images)
    #     interp = np.round(interp)