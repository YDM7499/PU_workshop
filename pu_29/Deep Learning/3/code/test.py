from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Input
from keras.layers import LSTM
from keras.models import Model
import numpy as np
import json
import codecs

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

#Test data
raw_text = 'average ma'

# integer encode sequences of characters
mapping_file_pointer = codecs.open('./mapping.json', 'r', 'utf-8')
mapping = json.loads(mapping_file_pointer.read())
print(mapping)

# integer encode line
encoded_seq = [mapping[char] for char in raw_text]
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = np.array(encoded_seq)
timestep = 10

x = encoded_seq	

x = np.array(x)
num_samples = 1

temp_x = np.zeros((num_samples, timestep, vocab_size))
for i in range(num_samples):
	int_temp_x = np.zeros((timestep, vocab_size))
	for j in range(timestep):
		int_temp_x[j,:] = to_categorical(x[i], num_classes=vocab_size)[j]
	temp_x[i, :, :] = int_temp_x
X_test = temp_x
#x.reshape((411, timestep, vocab_size))
#y.reshape((411, timestep, vocab_size))



inputs = Input(shape=(timestep, vocab_size), name='inputx')
x = LSTM(16, return_sequences=True)(inputs)
x = LSTM(vocab_size, return_sequences=True)(x)
model = Model(inputs=inputs, outputs=x)
model.summary()
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.load_weights('./model.h5')

out = model.predict(X_test)

inv_mapping = {v: k for k, v in mapping.items()}

def output_to_text(out):
	num_samples = out.shape[0]
	timestep = out.shape[1]
	vocab_size = out.shape[2]

	output_text = np.zeros((num_samples, timestep))
	output_text_str = ''
	for i in range(num_samples):
		for j in range(timestep):
			output_text[i, j] = np.argmax(out[i, j])
			output_text_str += str(inv_mapping[int(np.argmax(out[i, j]))])
	return output_text_str

output_text_str = output_to_text(out)
print(output_text_str)