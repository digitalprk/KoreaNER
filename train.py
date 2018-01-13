from import_data import load_corpus
from keras.models import Sequential
import numpy as np
from keras.layers import TimeDistributed, Dense, Activation, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing import sequence
from gensim.models.keyedvectors import KeyedVectors



vocab_dim = 200   # dimensionality of the word vectors
max_features = 20000
embedding_size = 128
hidden_size = 32


word_vectors = KeyedVectors.load('ko/ko.bin')
X, y = load_corpus()

all_text = [c for x in X for c in x]
words = list(set(all_text))
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}
labels = list(set([c for x in y for c in x]))
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

X = [[word2ind[c] for c in x] for x in X]
X = sequence.pad_sequences(X, maxlen=maxlen)    


max_label = max(label2ind.values()) + 1
y = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y = [[encode(c, max_label) for c in ey] for ey in y]
y = pad_sequences(y, maxlen=maxlen)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.3, random_state=42)

# prepare embedding matrix

nb_words = min(max_features, len(words) + 1)
embedding_matrix = np.zeros((nb_words, vocab_dim))
for word, i in word2ind.items():
    if i >= max_features:
        continue
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

model = Sequential()
model.add(Embedding(nb_words, vocab_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
#model.add(Embedding(input_dim=max_features, output_dim= 128,
#                    input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Raw test score:', score)


def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score(yh, pr)
print('Training accuracy:', accuracy_score(fyh, fpr))
print('Training confusion matrix:')
print(confusion_matrix(fyh, fpr))

pr = model.predict_classes(X_test)
yh = y_test.argmax(2)
fyh, fpr = score(yh, pr)
print('Testing accuracy:', accuracy_score(fyh, fpr))
print('Testing confusion matrix:')
print(confusion_matrix(fyh, fpr))