'''
    MLP from scratch as per the assignment in BT3041
    Author: Bharat K. Patil
'''

import numpy as np
from numpy import matlib
import pickle
from iteration_utilities import flatten
# from model import crit_net, crit_train, sigmoid, sigmoid_prime
# from atpbar import atpbar
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def feed_forward(inp, wts, biases, out):
    lay_inps = [] 
    lay_outs = []   

    # outputs of all layers for given input pattern or inpout pattern at t
    lay1_inp = np.matmul(wts[0], inp) - bias[0]
    temp_lay_out = sigmoid(lay1_inp)
    lay_inps.append(lay1_inp)
    lay_outs.append(temp_lay_out)
    for i in range(1,len(wts)):
        temp_lay_inp = np.matmul(wts[i], temp_lay_out) - bias[i]
        temp_lay_out = sigmoid(temp_lay_inp)
        lay_inps.append(temp_lay_inp)
        lay_outs.append(temp_lay_out)
    
    # calculating error (this changes if your data changes)
    delta = out - temp_lay_out

    return lay_inps, lay_outs, delta


def back(wts, bias, lay_inps, lay_outs, td, xi, eta):
    '''
        Back propagation used here is from sir's computational neuroscience course on NPTEL
    '''
    # find deltas for each layer
    wts.reverse(), lay_inps.reverse(), lay_outs.reverse(), bias.reverse()
    del_last = (sigmoid_prime(lay_inps[0]))*td
    deltas = []
    deltas.append(del_last)
    for ii in range(1,len(wts)):
        # print(ii)
        if ii == 1:
            temp1 = np.matlib.repmat(del_last, 1, wts[ii-1].shape[1])
            temp2 = wts[ii-1] * temp1
        else:
            temp1 = np.matlib.repmat(del_temp, 1, wts[ii-1].shape[1])
            temp2 = wts[ii-1] * temp1
        del_temp = (sigmoid_prime(lay_inps[ii])) * (np.sum(temp2, axis=0, keepdims=True).T)
        deltas.append(del_temp)
    # print(len(deltas))

    # update weights
    wts_updated = [] 
    for jj in range(len(wts)-1):
        # print(deltas[jj].shape)
        # print(lay_outs[jj+1].shape)
        del_w_temp = eta*np.matmul(deltas[jj], lay_outs[jj+1].T)
        wts[jj] = wts[jj] + del_w_temp
        wts_updated.append(wts[jj])
    wts[-1] = wts[-1] + eta*np.matmul(deltas[-1], xi.T)
    wts_updated.append(wts[-1])
    wts_updated.reverse()

    bias_updated = []
    for kk in range(len(bias)):
        bias[kk] = bias[kk] - eta*deltas[kk]
        bias_updated.append(bias[kk])
    bias_updated.reverse()

    return wts_updated, bias_updated

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(t):
    return sigmoid(t)*(1-sigmoid(t))

def accuracy(l1, l2):
    # uses a simple formula of (total matching)/(total samples) 
    pp = 0
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            pp+=1
    acc = pp/len(l1)
    return acc

# decides if you are training/testing
Train = True

# loading the data
with open("mnist_data.pkl", "rb") as f:
    d = pickle.load(f)
    f.close()
locals().update(d)

trainImages = np.asarray(trainImages)
testImages = np.asarray(testImages)

# preparing the data
labels_test = [] 
for i in testLabels:
    temp1 = np.zeros((10,1))
    temp1[i,0]=1
    labels_test.append(temp1)
labels_test = np.asarray(labels_test).reshape((len(testLabels),10))

labels_train = [] 
for j in trainLabels:
    temp2 = np.zeros((10,1))
    temp2[j,0]=1
    labels_train.append(temp2)
labels_train = np.asarray(labels_train).reshape((len(trainLabels),10))

trainImages = trainImages/np.amax(trainImages)
testImages = testImages/np.amax(testImages)


# initializing the model
wts = []
bias = []
error_total = []
acc_total = []
size = [784, 1000, 10] # model architecture
lr = 0.001 # learning rate
sz_in_train = trainImages.shape # size of train data
sz_in_test = testImages.shape   # size of testing data
decay = 2
epochs = 100

# initialize random wts
for i in range(len(size)-1):
    temp_wt = np.random.randn(size[i+1], size[i])
    temp_bias = np.random.randn(size[i+1], 1)
    wts.append(temp_wt)
    bias.append(temp_bias)


if Train:
    for j in range(epochs):
        error = 0
        lab_pred = []
        
        if (j%10 == 0):
            print(j)
            # lr = lr/decay

        for i in range(sz_in_train[0]):
            # feed forward
            inp = trainImages[i,:].reshape(trainImages.shape[1], 1)
            out = labels_train[i,:].reshape(labels_train.shape[1], 1)

            lays_inp, lays_out, err1 = feed_forward(inp, wts, bias, out)

            # back propagation
            wts, bias = back(wts, bias, lays_inp, lays_out, err1, inp, lr)
            error += np.sum(err1**2)/2
        error_total.append(error)
        
        # calculate the labels and the output from the final layer
        a = np.where(lays_out[-1] == np.amax(lays_out[-1]))
        lab_pred.append(a[0][0])
        acc_total.append(accuracy(lab_pred, trainLabels))

    
    p = {'wts':wts, 'error_total': error_total,'acc_total': acc_total, 'bias': bias, 'epochs':epochs, 'lr':lr}
    with open('trained_model.pk1', 'wb') as w:
        pickle.dump(p, w)
        w.close()

    # plot the loss curve and the accuracy curve
    plt.plot(range(len(error_total)), error_total)
    plt.title("loss curve for training data")
    plt.show()
    plt.plot(range(len(acc_total)), acc_total)
    plt.show()

else: # testing
    # load model
    with open('trained_model.pk1', 'rb') as w2:
        p = pickle.load(w2)
        w2.close()
    locals().update(p)
    
    output_labels = []
    y_train_pred= []
    y_test_pred= []

    for i in range(sz_in_train[0]):
        # feed forward
        inp1 = trainImages[i,:].reshape(trainImages.shape[1], 1)
        out1 = labels_train[i,:].reshape(labels_train.shape[1], 1)
        # print(out)
        lays_inp1, lays_out1, err1 = feed_forward(inp1, wts, bias, out1)
        y_train_pred.append(lays_out1[-1])

    # uncomment the following for results on testing data
'''
    for i in range(sz_in_test[0]):
        # feed forward
        inp2 = testImages[i,:].reshape(testImages.shape[1], 1)
        out2 = labels_test[i,:].reshape(labels_test.shape[1], 1)
        # print(out)
        lays_inp, lays_out, err1 = feed_forward(inp2, wts, bias, out2)
        y_test_pred.append(lays_out[-1])

    y_train = labels_train
    y_test = labels_test
    y_train_pred = np.asarray(y_train_pred).reshape(800,10)
    y_test_pred = np.asarray(y_test_pred).reshape(200,10)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))
    # print(output_labels)
'''