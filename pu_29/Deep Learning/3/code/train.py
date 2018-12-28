from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input
from keras.layers import LSTM
from keras.models import Model
from keras.callbacks import LearningRateScheduler
import numpy as np
import json
import codecs
import math

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'data.txt'
raw_text = load_doc(in_filename)

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

# integer encode line
encoded_seq = [mapping[char] for char in raw_text]

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = np.array(encoded_seq)
timestep = 10

x = []
y = []
for i in range(len(encoded_seq)-11):
	x.append(encoded_seq[i:timestep + i])
	y.append(encoded_seq[i + 1:timestep + i+ 1])

x = np.array(x)
y = np.array(y)
num_samples = x.shape[0]

temp_x = np.zeros((num_samples, timestep, vocab_size))
for i in range(num_samples):
	int_temp_x = np.zeros((timestep, vocab_size))
	for j in range(timestep):
		int_temp_x[j,:] = to_categorical(x[i], num_classes=vocab_size)[j]
	temp_x[i, :, :] = int_temp_x

temp_y = np.zeros((num_samples, timestep, vocab_size))
for i in range(num_samples):
	int_temp_y = np.zeros((timestep, vocab_size))
	for j in range(timestep):
		int_temp_y[j,:] = to_categorical(y[i], num_classes=vocab_size)[j]
	temp_y[i, :, :] = int_temp_y

#x = np.array(temp_x)
#y = np.array(temp_y)

X = temp_x
Y = temp_y
print(x.shape, y.shape)
#x.reshape((411, timestep, vocab_size))
#y.reshape((411, timestep, vocab_size))

def step_decay(epoch):
	initial_lrate = 1e-3
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

lrate = LearningRateScheduler(step_decay)

inputs = Input(shape=(timestep, vocab_size), name='inputx')
x = LSTM(16, return_sequences=True)(inputs)
x = LSTM(vocab_size, return_sequences=True)(x)
model = Model(inputs=inputs, outputs=x)
model.summary()
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, Y, epochs=1, verbose=2, callbacks=[lrate])

# save the model to file
model.save('model.h5')

# save the mapping

mapping_file_pointer = codecs.open('./mapping.json', 'w', 'utf-8')
mapping_file_pointer.write(json.dumps(mapping))
