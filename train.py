from import_data import load_corpus
import numpy as np
from keras.layers import TimeDistributed, Dense, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.models import Model
from crf import ChainCRF
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.layers import Input, Dropout, Reshape, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
from collections import Counter
from itertools import product
import functools
from jamo import decompose_character

vocab_dim = 200
embedding_size = 128
hidden_size = 32
nb_filters = 10
char_embedding_size = 15
jamo_embedding_size = 10
batch_size = 32

word_vectors = KeyedVectors.load('ko/ko.bin')
X, y, pos_tags = load_corpus()

all_text = [c for x in X for c in x]
labels = list(set([c for x in y for c in x]))
max_word_len = max([len(c) for x in X for c in x])
max_jamo_len = max_word_len * 3

words = list(set(all_text))
words.append('UNKWRD')

chars = list(set([char for word in words for char in word]))
chars.append('UNKCHR')

jamos = list(set([jamo for word in words for char in word for jamo in decompose_character(char)]))
jamos.append('UNKJAM')

tags = list(set([tag for list_pos in pos_tags for tag in list_pos]))
tags.append('UNKPOS')

char2ind = {char: index for index, char in enumerate(chars)}
ind2char = {index: char for index, char in enumerate(chars)}

word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}

jamo2ind = {jamo: (index + 1) for index, jamo in enumerate(jamos)}
ind2jamo = {(index + 1): jamo for index, jamo in enumerate(jamos)}

label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

pos2ind = {pos: (index + 1) for index, pos in enumerate(tags)}
ind2pos = {(index + 1): pos for index, pos in enumerate(tags)}



out_size = len(label2ind) + 1
lengths = [len(x) for x in X]
print('Input sequence length range: ', max(lengths), min(lengths))
maxlen = max([len(x) for x in X])

def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

# Create a 3D Matrix num_of_sentences * words (padded to max num of words) * chars (padded to max num of chars)

X_char = [sequence.pad_sequences([[char2ind[char] for char in word] for word in x], maxlen = max_word_len) for x in X]
X_char = sequence.pad_sequences(X_char, maxlen=maxlen)

# Create a matrix for jamos

X_jamo = [sequence.pad_sequences([[jamo2ind[jamo] for char in word for jamo in decompose_character(char)] for word in x], maxlen = max_jamo_len) for x in X]
X_jamo = sequence.pad_sequences(X_jamo, maxlen=maxlen)

# Create a matrix for POS Tags

max_pos = max(pos2ind.values()) + 1
X_pos = [[0] * (maxlen - len(ey)) + [pos2ind[c] for c in ey] for ey in pos_tags]
X_pos = [[encode(c, max_pos) for c in ey] for ey in X_pos]
X_pos = pad_sequences(X_pos, maxlen=maxlen)

# Create a matrix with word index

X = [[word2ind[c] for c in x] for x in X]
X = sequence.pad_sequences(X, maxlen=maxlen)

# Convert labels to one-hot vectors

max_label = max(label2ind.values()) + 1
y = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y = [[encode(c, max_label) for c in ey] for ey in y]
y = pad_sequences(y, maxlen=maxlen)


# train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.3, random_state=42)

X_char_train, X_char_test, y_train, y_test = train_test_split(X_char, y,
                                               test_size=0.3, random_state=42)

X_jamo_train, X_jamo_test, y_train, y_test = train_test_split(X_jamo, y,
                                               test_size=0.3, random_state=42)

X_pos_train, X_pos_test, y_train, y_test = train_test_split(X_pos, y,
                                               test_size=0.3, random_state=42)
# class weights

frequency = [list(array).index(1) for arrays in y_test for array in arrays]
frequency = dict(Counter(frequency))
frequency[0] = 0
total = sum([frequency[k] for k in frequency])
frequency = {k: frequency[k] / total for k in frequency}
category_weights = np.zeros(out_size)
for f in frequency:
    category_weights[f] = frequency[f]

weights = []

for sample in y_train:
    current_weight = []
    for line in sample:
        current_weight.append(frequency[list(line).index(1)])
    weights.append(current_weight)
weights = np.array(weights)

"""
for i, sample in enumerate(y_train):
    matrix_weight = np.array([])
    for j, line in enumerate(sample):
        matrix_weight.concatenate(line * category_weights)
    weights.append(matrix_weight)
"""
    

# Prepare word embedding matrix from pre-trained vectors

embedding_matrix = np.zeros((len(words) + 1, vocab_dim))
for word, i in word2ind.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

word_idx = Input(batch_shape=(None, maxlen), dtype='int32')
word_embeddings = Embedding(len(words) + 1, vocab_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True)(word_idx)

# CNN Layers for character embeddings

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

# Single CNN layer for jamo embeddings

def Single_CNN(seq_length, length, input_size, feature_maps, kernels, x): #testing with single layer
    kernel = 3
    reduced_l = length - kernel + 1
    conv = Conv2D(50, (1, kernel), activation='tanh', data_format="channels_last")(x)
    maxp = MaxPooling2D((1, reduced_l), data_format="channels_last")(conv)
    x = Reshape((seq_length, 50))(maxp)
    return x  

# Create char embeddings

char_idx = Input(batch_shape=(None, maxlen, max_word_len), dtype='int32')
char_embeddings = TimeDistributed(Embedding(len(char2ind) + 1, char_embedding_size))(char_idx)
cnn = Single_CNN(maxlen, max_word_len, char_embedding_size, feature_maps, kernels, char_embeddings)

# Create jamo embeddings

jamo_idx = Input(batch_shape=(None, maxlen, max_jamo_len), dtype='int32')
jamo_embeddings = TimeDistributed(Embedding(len(jamo2ind) + 1, jamo_embedding_size))(jamo_idx)
jamo_cnn = Single_CNN(maxlen, max_jamo_len, jamo_embedding_size, feature_maps, kernels, jamo_embeddings)

# Handle POS Tags

pos_idx = Input(batch_shape=(None, maxlen, max_pos), dtype='float32')

# Concatenate character embeddings and word embeddings

x = Concatenate()([cnn, word_embeddings, jamo_cnn, pos_idx])
inputs = [char_idx, word_idx, jamo_idx, pos_idx]

# Model is Bi-LSTM with a CRF Layer

x = BatchNormalization()(x)
x = Bidirectional(LSTM(hidden_size, return_sequences=True))(x)
"""
output = TimeDistributed(Dense(out_size, activation='softmax'))(x)

loss = WeightedCategoricalCrossEntropy(frequency)

model = Model(inputs = inputs, outputs = output)
model.compile(loss = loss, optimizer='adam') #loss='categorical_crossentropy', optimizer='adam')

"""
output = Dense(out_size)(x)
crf = ChainCRF()
crf_output = crf(output)

model = Model(inputs = inputs, outputs = crf_output)
model.summary()
model.compile(loss=crf.loss, optimizer='adam', sample_weight_mode='temporal')


early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode = 'auto')

model.fit([X_char_train, X_train, X_jamo_train, X_pos_train], y_train, batch_size=batch_size, epochs=15,
          validation_data=([X_char_test, X_test, X_jamo_test, X_pos_test], y_test), callbacks = [early_stop], sample_weight = weights)
score = model.evaluate([X_char_test, X_test, X_jamo_test, X_pos_test], y_test, batch_size=batch_size)
print('Raw test score:', score)


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

pr = model.predict([X_char_train, X_train, X_jamo_train, X_pos_train])
yh = y_train.argmax(2)
pr = pr.argmax(2)
fyh, fpr = score(yh, pr)
print('Training accuracy:', accuracy_score(fyh, fpr))
print(classification_report(fyh, fpr))

pr = model.predict([X_char_test, X_test, X_jamo_test, X_pos_test])
yh = y_test.argmax(2)
pr = pr.argmax(2)
fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print(classification_report(fyh, fpr))