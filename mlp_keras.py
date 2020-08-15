'''
    MLP using keras model as per the assignment in BT3041
    Author: Bharat K. Patil
'''

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
import pickle, os, matplotlib
import numpy as np
import sklearn, timeit
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
from sklearn.metrics import r2_score

# uploading data
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
X_train = trainImages
y_train = labels_train
X_test = testImages
y_test = labels_test

# model specifications
Train = True
LR = [0.000001] #, 0.001, 0.0001]
output_size = 10
epochs = 10
nodes = [100]
act = 'sigmoid'

if Train:
    # initialize keras model
    model = Sequential()

    model.add(Dense(nodes[0], input_shape = (784,), activation = act)) #1
    # model.add(Dense(nodes[1], activation = act)) #1
    # model.add(Dense(nodes[2], activation = act)) #1
    model.add(Dense(output_size, activation= act)) #5

    model.compile(Adam(lr=i), loss='mean_squared_error')

    #Fits model
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = 0.1)
    history_dict=history.history

    #Plots model's training cost/loss and model's validation split cost/loss
    loss_values = history_dict['loss']
    val_loss_values=history_dict['val_loss']
    plt.figure()
    plt.plot(loss_values,'bo',label='training loss')
    plt.plot(val_loss_values,'r',label='val training loss')
    plt.savefig("loss curve.png")
    plt.show()

    model.save("model.h5")

else: 
    modelx = load_model("model.h5")

    y_train_pred = modelx.predict(X_train)
    y_test_pred = modelx.predict(X_test)

    # Calculates and prints r2 score of training and testing data
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))

    ff = open('result.txt', 'w+')
    # for _ in range(2):
    ff.write("The R2 score on the Train set is: " + str(r2_score(y_train, y_train_pred)) +"\n")
    ff.write("The R2 score on the Test set is: " + str(r2_score(y_test, y_test_pred)))
    ff.close()