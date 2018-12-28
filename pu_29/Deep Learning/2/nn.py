from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import tensorboard
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import codecs

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

"""
print(x_train[0])
print(y_train[0])

plt.imshow(x_train[0])
plt.show()
"""

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Plot images
# Convert data into 2d and get numpy arrays
temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())
x_train = np.array(temp_x_train)

temp_x_test = []
for i in range(x_test.shape[0]):
	temp_x_test.append(x_test[i].ravel())
x_test = np.array(temp_x_test)
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

#Define model
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=784, name='input'))
model.add(Dense(128, activation='relu', name='hidden1'))
model.add(Dense(64, activation='relu', name='hidden2'))
model.add(Dense(10, activation='softmax', name='hidden3'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))#, callbacks=[tensorboard_callback])

#tensorboard.tensorboard_service(model)