from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils, generic_utils
from six.moves import range
import numpy as np
import scipy as sp
from keras import backend as K  
import random
import scipy.io
from scipy.stats import mode
# custom module for cnn model
from cnn_architecture2 import *

import argparse


#parameters
batch_size = 128
nb_classes = 10
nb_epoch = 50 #use a large number of epochs
img_rows, img_cols = 28, 28 # input image dimensions
input_shape = (img_rows, img_cols, 1)

score=0
all_accuracy = 0
acquisition_iterations = 98
dropout_iterations = 100 #use a large number of dropout iterations
num_of_queries = 10


# Training settings
parser = argparse.ArgumentParser(description='MNIST AL')
parser.add_argument('--trial_number', type=int, default=0, metavar='N',
                    help='trial number for given acquisition function (default: 0)')
parser.add_argument('--acquisition_function', type=str, default='RANDOM', metavar='N',
                    help='type of acquisition. Options are: RANDOM, ENTROPY, VAR_RATIO')
args = parser.parse_args()

print("Training with the following acquisition function: ", args.acquisition_function)
print("Training for trial #: ", args.trial_number)

# the data, shuffled and split between train and test sets
(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

# reshape to (X, 28, 28, 1)
X_train_All = X_train_All.reshape(X_train_All.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

random_split = np.asarray(random.sample(range(0, X_train_All.shape[0]), X_train_All.shape[0]))

X_train_All = X_train_All[random_split]
y_train_All = y_train_All[random_split]

X_Pool = X_train_All[20000:60000]
y_Pool = y_train_All[20000:60000]

X_train_All = X_train_All[0:10000]
y_train_All = y_train_All[0:10000]


# extract 2 instances of each digit type and add to training set
X_train, y_train = np.array([], dtype=np.int64), np.array([], dtype=np.int64)
for index in range(10):
    idx = np.array(np.where(y_train_All==index)).T
    idx = idx[0:2, 0]
    X = X_train_All[idx, :, :, :]
    y = y_train_All[idx]
    
    X_train = np.concatenate((X_train, X), axis=0 ) if X_train.size else X
    y_train = np.concatenate((y_train, y), axis=0 ) if y_train.size else y


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')

print('Distribution of Training Classes:', np.bincount(y_train))

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_Pool = X_Pool.astype('float32')
X_train /= 255
X_Pool /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)


Pool_Train_Acc = np.zeros(shape=(nb_epoch, 1)) 

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=0)
Train_Result_Optimizer = hist.history
Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
Train_Acc = np.array([Train_Acc]).T
Pool_Train_Acc = Train_Acc

print('Evaluating Test Accuracy Without Acquisition')
score, acc = model.evaluate(X_test, Y_test, verbose=0)
all_accuracy = acc

print('Starting Active Learning in Experiment ')

nb_MC_samples = 100
MC_output = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

for i in range(acquisition_iterations):
    print('ACQUISITION ITERATION ' + str(i+1) + ' of ' + str(acquisition_iterations))

    if (args.acquisition_function == 'RANDOM'):
        acquired_index = np.asarray(random.sample(range(0, X_Pool.shape[0]), num_of_queries))
    #other methods require MCDropout
    else:
        pool_subset_count = 2000
        pool_subset_random_index = np.asarray(random.sample(range(0, X_Pool.shape[0]), pool_subset_count))
        X_Pool_subset = X_Pool[pool_subset_random_index]
        y_Pool_subset = y_Pool[pool_subset_random_index]

        print('Dropout over Pool of Unlabeled Data')
        
        # Var ratio active learning acquisition function
        # get MC_Samples
        learning_phase = True  # use dropout at test time
        MC_samples = [MC_output([X_Pool_subset, learning_phase])[0] for _ in range(dropout_iterations)]
        MC_samples = np.array(MC_samples)  # [#samples x batch size x #classes]    
        s = np.mean(MC_samples, axis=0)

        if (args.acquisition_function == 'ENTROPY'):
            entropy = np.sum(np.multiply(s, np.log(s)), axis=1) 
            acquired_index = np.argsort(entropy)[:num_of_queries]
        elif (args.acquisition_function == 'VAR_RATIO'):
            acquired_index = np.argsort(np.max(s, axis=1))[:num_of_queries]

        
    acquired_X = X_Pool_subset[acquired_index] 
    acquired_Y = y_Pool_subset[acquired_index]	

    # Remove the acquired data from the unlabeled Pool
    X_Pool = np.delete(X_Pool, (pool_subset_random_index[acquired_index]), axis=0)
    y_Pool = np.delete(y_Pool, (pool_subset_random_index[acquired_index]), axis=0)

    print('Acquired Points added to the training set')
    X_train = np.concatenate((X_train, acquired_X), axis=0)
    y_train = np.concatenate((y_train, acquired_Y), axis=0)
    print('Train Data size: ' + str(X_train.shape))  
    print('Unlabeled Pool size: ' + str(X_Pool.shape))

    print('Train Again with the added points')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=0)
    
    Train_Result_Optimizer = hist.history
    Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
    Train_Acc = np.array([Train_Acc]).T
    Pool_Train_Acc = np.append(Pool_Train_Acc, Train_Acc, axis=1)	


    print('Evaluate Model Test Accuracy after training')

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    all_accuracy = np.append(all_accuracy, acc)
    print()


print('Storing Accuracy Values over experiments')
test_str = './results/' + args.acquisition_function + '_test_acc_' + args.trial_number + '.npy'
train_str =  './results/' + args.acquisition_function + '_train_acc_' + args.trial_number + '.npy'

np.save(test_str, all_accuracy)
np.save(train_str, Pool_Train_Acc)






