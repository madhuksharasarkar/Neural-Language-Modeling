from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional, GRU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import text_to_word_sequence

import numpy as np
import random
import os
import codecs
from six.moves import cPickle
import matplotlib.pyplot as plt
import collections

save_dir = 'save'
data_dir = 'dickens'

file_list = ["pg23344","924-0","807-0","pg1392","pg1407","1467-0"]

vocab_file = os.path.join(save_dir, "words_vocab.pkl")
sequences_step = 1

def remove_gutenberg_text(content):
    paragraphs = (p for p in content.split('\n') if p != '')
    include = False
    START_PREFIX = '***START OF'
    END_PREFIX = '***END OF'
    
    non_gutenberg_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph[:len(END_PREFIX)] == END_PREFIX:
            include = False
        
        if include:
            non_gutenberg_paragraphs.append(paragraph)
        
        if paragraph[:len(START_PREFIX)] == START_PREFIX:
            include = True
    
    return '\n'.join(non_gutenberg_paragraphs)

word_list = []
for file_name in file_list:
    input_file = os.path.join(data_dir, file_name + ".txt")
    print (input_file)
    
    with codecs.open(input_file, encoding="utf8") as f:
        data = f.read()
        data = remove_gutenberg_text(data)
        wordlist = text_to_word_sequence(data, filters='!"#$%&()*+,''-./:;<=>?@[\\]^_`{|}~\t\n\r')
        word_list.extend(wordlist)
        length = len(wordlist)
        print (length)

word_counts = collections.Counter(word_list)

vocabulary_inv = [x for x in word_list]
vocabulary_inv = set(sorted(vocabulary_inv))

vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

vocab_size = len(words)

with open(os.path.join(vocab_file), 'wb') as f:
    cPickle.dump((words, vocab, vocabulary_inv), f)

seq_length = 30
sequences = []
next_words = []
for i in range(0, len(word_list) - seq_length, sequences_step):
    sequences.append(word_list[i: i + seq_length])
    next_words.append(word_list[i + seq_length])

X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
for i, sentence in enumerate(sequences):
    for t, word in enumerate(sentence):
        X[i, t, vocab[word]] = 1
    y[i, vocab[next_words[i]]] = 1

def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

###############################################################################
############################### Model 1: LSTM #################################
###############################################################################

rnn_size = 512
batch_size = 30
num_epochs = 10
learning_rate = 0.001

model = Sequential()
model.add(LSTM(rnn_size, activation="relu", input_shape=(seq_length, vocab_size)))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print (model.summary())

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks=[ModelCheckpoint(filepath=save_dir + "/" + 'my_model_lstm.hdf5',\
                          monitor='loss', verbose=0, mode='auto', period=1)]

hist = model.fit(X, y, batch_size=batch_size, shuffle=True, epochs=num_epochs, callbacks=callbacks)

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss'], loc='upper right')
plt.show()

model.save(save_dir + "/" + 'my_model_lstm.h5')

vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

model = load_model(save_dir + "/" + 'my_model_lstm.h5')



words_number = 50
start_index = random.randint(0, len(wordlist) - seq_length - 1)

generated = ''
sentence = word_list[start_index: start_index + seq_length]
generated += ' '.join(sentence)

for i in range(words_number):
    
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = list(vocabulary_inv)[next_index]
    
    generated += " " + next_word
    
    sentence = sentence[1:] + [next_word]

print(generated)

###############################################################################
####################### Model 2: Bidirectional LSTM ###########################
###############################################################################

rnn_size = 256
batch_size = 30
num_epochs = 10
learning_rate = 0.001

model = Sequential()
model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
model.add(Dropout(0.2))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print (model.summary())

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks=[ModelCheckpoint(filepath=save_dir + "/" + 'my_model_bidirectional_lstm.hdf5',\
                          monitor='loss', verbose=0, mode='auto', period=1)]

hist = model.fit(X, y, batch_size=batch_size, shuffle=True, epochs=num_epochs, callbacks=callbacks)

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss'], loc='upper right')
plt.show()

model.save(save_dir + "/" + 'my_model_bidirectional_lstm.hdf5')

vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

model = load_model(save_dir + "/" + 'my_model_bidirectional_lstm.hdf5')

words_number = 50
start_index = random.randint(0, len(wordlist) - seq_length - 1)

generated = ''
sentence = word_list[start_index: start_index + seq_length]
generated += ' '.join(sentence)

for i in range(words_number):
    
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = list(vocabulary_inv)[next_index]
    
    generated += " " + next_word
    
    sentence = sentence[1:] + [next_word]

print(generated)

###############################################################################
################################# Model 3: GRU ################################
###############################################################################

rnn_size = 512
batch_size = 30
num_epochs = 15
learning_rate = 0.001

model = Sequential()
model.add(GRU(rnn_size, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
print (model.summary())

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks=[ModelCheckpoint(filepath=save_dir + "/" + 'my_model_gru.hdf5',\
                          monitor='loss', verbose=0, mode='auto', period=1)]

hist = model.fit(X, y, batch_size=batch_size, shuffle=True, epochs=num_epochs, callbacks=callbacks)

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss'], loc='upper right')
plt.show()

model.save(save_dir + "/" + 'my_model_gru.h5')

vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)

model = load_model(save_dir + "/" + 'my_model_gru.h5')

words_number = 50
start_index = random.randint(0, len(wordlist) - seq_length - 1)

generated = ''
sentence = word_list[start_index: start_index + seq_length]
generated += ' '.join(sentence)

for i in range(words_number):
    
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.33)
    next_word = list(vocabulary_inv)[next_index]
    
    generated += " " + next_word
    
    sentence = sentence[1:] + [next_word]

print(generated)