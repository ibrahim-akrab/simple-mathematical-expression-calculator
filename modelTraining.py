import numpy as np
import matplotlib.pyplot as plt

# load the dataset
digits = np.load('dataset/digits.npy').T
classes = np.load('dataset/classes.npy').T

#
#print(digits.shape)
#print(classes.shape)
#from random import randrange
#for _ in range(10):
#    index = randrange(1756)
#    plt.imshow(digits[index].reshape((16,16)))
#    plt.show()
#    print(np.where(classes[index] == 1)[0][0])
#


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits, classes, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

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

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 200)


# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype('int64')


# calculating metrics for the neural network model using sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

y_test = y_test.argmax(axis=1)
y_pred = y_pred.argmax(axis=1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, y_pred, average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, y_pred, average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, y_pred, average='micro')
print('F1 score: %f' % f1)
# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

model.save('model.h5')