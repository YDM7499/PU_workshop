from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

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
lines = raw_text.split('\n')

# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

# separate into input and output
sequences = np.array(sequences)

X, y = sequences[:len(sequences) - 1], sequences[1:len(sequences)]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
sequences = [to_categorical(y_, num_classes=vocab_size) for y_ in y]
y = np.array(sequences)
print(X)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(None, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')

# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))