import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

plt.imshow(x_train[0], cmap='gray')
plt.show()

temp_x_train = []
for i in range(x_train.shape[0]):
	temp_x_train.append(x_train[i].ravel())

x_train = np.array(temp_x_train)

D = 784
K = 10
# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength
iteration = 10

# gradient descent loop
num_examples = x_train.shape[0]

for i in range(iteration):
	# evaluate class scores, [N x K]
	scores = np.dot(x_train, W) + b

	# compute the class probabilities
	exp_scores = np.exp(scores)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

	# compute the loss: average cross-entropy loss and regularization
	correct_logprobs = -np.log(probs[range(num_examples),y_train])
	data_loss = np.sum(correct_logprobs)/num_examples
	reg_loss = 0.5*reg*np.sum(W*W)
	loss = data_loss + reg_loss
	if i % 10 == 0:
		print("iteration %d: loss %f" % (i, loss))

	# compute the gradient on scores
	dscores = probs
	dscores[range(num_examples),y_train] -= 1
	dscores /= num_examples

	# backpropagate the gradient to the parameters (W,b)
	dW = np.dot(x_train.T, dscores)
	db = np.sum(dscores, axis=0, keepdims=True)

	dW += reg*W # regularization gradient

	# perform a parameter update
	W += -step_size * dW
	b += -step_size * db


	initial_lrate = 0.1
	k = 0.1
	step_size = initial_lrate * np.exp(-k*(i + 1))
