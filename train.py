from import_data import load_corpus
from keras.models import Sequential
import numpy as np
from keras.layers import TimeDistributed, Dense, Activation, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing import sequence
from keras.models import Model
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report
#from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D, Conv2D
#from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute, Reshape, Input
from keras.layers import Input, Dropout, Reshape, Concatenate, Conv2D, MaxPooling2D, BatchNormalization



vocab_dim = 200   # dimensionality of the word vectors
max_features = 20000
embedding_size = 128
hidden_size = 32
nb_filters = 10
char_embedding_size = 15

word_vectors = KeyedVectors.load('ko/ko.bin')
X, y = load_corpus()

all_text = [c for x in X for c in x]
labels = list(set([c for x in y for c in x]))
max_word_len = max([len(c) for x in X for c in x])

words = list(set(all_text))
chars = list(set([char for word in words for char in word]))

char2ind = {char: index for index, char in enumerate(chars)}
ind2char = {index: char for index, char in enumerate(chars)}

word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}

label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

out_size = len(label2ind) + 1
lengths = [len(x) for x in X]
print('Input sequence length range: ', max(lengths), min(lengths))
maxlen = max([len(x) for x in X])

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

# prepare char embeddings

# Create a 3D Matrix num_of_sentences * words (padded to max num of words) * chars (padded to max num of chars)
X_char = [sequence.pad_sequences([[char2ind[char] for char in word] for word in x], maxlen = max_word_len) for x in X]
X_char = sequence.pad_sequences(X_char, maxlen=maxlen)
#X_char = X_char.reshape(len(X_char),maxlen*max_word_len)

"""
X_char = []
for sentence in X:
    for word in sentence:
        word_chars = []
        for character in word:
            word_chars.append(char2ind[character])
            
        X_char.append(word_chars)
X_char = sequence.pad_sequences(X_char, maxlen = max_word_len)
"""          

X = [[word2ind[c] for c in x] for x in X]
X = sequence.pad_sequences(X, maxlen=maxlen)

#X = [X, X_char]

#x = 0 / 0

max_label = max(label2ind.values()) + 1
y = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y = [[encode(c, max_label) for c in ey] for ey in y]
y = pad_sequences(y, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.3, random_state=42)

X_char_train, X_char_test, y_train, y_test = train_test_split(X_char, y,
                                               test_size=0.3, random_state=42)

# prepare word embedding matrix

embedding_matrix = np.zeros((len(words) + 1, vocab_dim))
for word, i in word2ind.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

word_idx = Input(batch_shape=(None, maxlen), dtype='int32')
word_embeddings = Embedding(len(words) + 1, vocab_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True)(word_idx)

# prepare char embedding

feature_maps = [50,100,150,200,200,200,200]
kernels = [1,2,3,4,5,6,7]

def CNN(seq_length, length, input_size, feature_maps, kernels, x):
    
    concat_input = []
    for feature_map, kernel in zip(feature_maps, kernels):
        reduced_l = length - kernel + 1
        conv = Conv2D(feature_map, (1, kernel), activation='tanh', data_format="channels_last")(x)
        maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
        concat_input.append(maxp)

    x = Concatenate()(concat_input)
    x = Reshape((seq_length, sum(feature_maps)))(x)
    return x

char_idx = Input(batch_shape=(None, maxlen, max_word_len), dtype='int32')
char_embeddings = TimeDistributed(Embedding(len(char2ind) + 1, char_embedding_size))(char_idx)
cnn = CNN(maxlen, max_word_len, char_embedding_size, feature_maps, kernels, char_embeddings)


#word_embeddings = Embedding(len(words) + 1, vocab_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True)

x = Concatenate()([cnn, word_embeddings])
inputs = [char_idx, word_idx]

x = BatchNormalization()(x)
x = Bidirectional(LSTM(hidden_size, return_sequences=True))(x)
output = TimeDistributed(Dense(out_size, activation='softmax'))(x)
#x = TimeDistributed(Dense(out_size))(x)
#output = Activation('softmax')(x)
#output = TimeDistributed(Dense(len(words) + 1, activation='softmax'))(x)
model = Model(inputs = inputs, outputs = output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')

"""
model = Sequential()
model.add(Embedding(len(words) + 1, vocab_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
#model.add(Embedding(input_dim=max_features, output_dim= 128,
#                    input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
"""

batch_size = 32
model.fit([X_char_train, X_train], y_train, batch_size=batch_size, epochs=5,
          validation_data=([X_char_test, X_test], y_test))
score = model.evaluate([X_char_test, X_test], y_test, batch_size=batch_size)
print('Raw test score:', score)


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

pr = model.predict([X_char_train, X_train])
yh = y_train.argmax(2)
pr = pr.argmax(2)
fyh, fpr = score(yh, pr)
print('Training accuracy:', accuracy_score(fyh, fpr))
print(classification_report(fyh, fpr))

pr = model.predict([X_char_test, X_test])
yh = y_test.argmax(2)
pr = pr.argmax(2)
fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print(classification_report(fyh, fpr))