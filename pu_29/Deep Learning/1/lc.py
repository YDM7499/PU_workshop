import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import json
import codecs

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)

# Normalize data
x_train = x_train.astype('float32') / 255

# Plot images
# Convert data into 2d and get numpy arrays
temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())
x_train = np.array(temp_x_train)

# Initilize dimension, class_no, Step size, regularization constant, and iteration
D = 784
K = 10
step_size = 1e-0
reg = 1e-3
iteration = 10
num_samples = x_train.shape[0]

# Init weights and bias
W = 0.01 * np.random.randn(784, 10)
B = 0.01 * np.random.randn(1, 10)

# Within loop forward and back pass
all_loss = []
for i in range(iteration):
	h = np.dot(x_train, W) + B
	
	# Get scores
	# compute the class probabilities
	# compute the loss: average cross-entropy loss and regularization
	exp_scores = np.exp(h)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

	# compute the loss: average cross-entropy loss and regularization
	correct_logprobs = -np.log(probs[range(num_samples),y_train])
	data_loss = np.sum(correct_logprobs)/num_samples
	reg_loss = 0.5*reg*np.sum(W*W)
	loss = data_loss + reg_loss
	print('loss at iteration ', i, ': ', loss)
	# compute the gradient on scores
	dh = probs
	dh[range(num_samples),y_train] -= 1
	dh /= num_samples

	dW = np.dot(x_train.T, dh)
	dB = np.sum(dh, axis = 0, keepdims=True)

	W -= step_size*dW
	B -= step_size*dB

	initial_lrate = 1
	k = 0.1
	step_size = initial_lrate * np.exp(-k*(i + 1))
	
	# backpropagate the gradient to the parameters (W,b)
	# perform a parameter update
	# LR decay

	all_loss.append(loss)

#plt.plot(range(100000), all_loss)
#plt.show()

writing_file_pointer = codecs.open('./weight.json', mode = 'w')
dict_ = {'w': W, 'b': B}
writing_file_pointer.write(str(dict_))