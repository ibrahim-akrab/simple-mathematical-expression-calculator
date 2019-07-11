# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load the dataset
digits = np.load('dataset/digits.npy').T
classes = np.load('dataset/classes.npy').T

# define baseline model
def baseline_model():
    # Initialising the NN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(units=128, activation='softplus', input_dim=16*16))
    
    # Adding the second hidden layer
    model.add(Dense(units=32, activation='softplus'))
    
    # Adding the output layer
    model.add(Dense(units = 12, activation = 'softmax'))
    
    # Compiling the NN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5)
#global_model = baseline_model()
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, digits, classes, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
