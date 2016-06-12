'''mple script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
import preprocess
import sys

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']
continueTraining = False 

try: 
    textR = preprocess.getReports(REPORT_FILES)
    text = ""
    #text = open("./shakespeareEdit.txt").read().lower()

    testText = ""
    fullText = ""
    totalTextLen = 5000000
    maxlen = 20 
    allText = []
    for i in xrange(len(textR)):
	text += textR[i][:len(textR[i])/2]
	testText += textR[i][len(textR[i])/2:]
	fullText += textR[i]
    chars = set(fullText)
    fullText = ""
    textText = ""

    for i in range(0,len(text) - totalTextLen,totalTextLen):
	allText.append(text[i:i+totalTextLen])
    #text=allText[0]
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(text))

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# build the model: 2 stacked LSTM
print('Build model...')
if not continueTraining:
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
else:
    m = model_from_json(open('./model_files/reports.rnn_char_architecture.json').read())
    m.load_weights('./model_files/reports.rnn_char_weights.h5')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


for iteration in range(1, 60):
    for curText in range(1,len(allText)):
	# cut the text in semi-redundant sequences of maxlen characters
	print('Preprocessing')
	step = 3
	sentences = []
	next_chars = []
	for i in range(0, len(allText[curText]) - maxlen, step):
		sentences.append(allText[curText][i: i + maxlen])
		next_chars.append(allText[curText][i + maxlen])

	X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			X[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	print('Finished Preprocessing')

	# train the model, output generated text after each iteration
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=256, nb_epoch=1)

	start_index = random.randint(0, totalTextLen - maxlen - 1)

	for diversity in [0.2, 0.5, 1.0, 1.2]:
	    print()
	    print('----- diversity:', diversity)

	    generated = ''
	    sentence = allText[curText][start_index: start_index + maxlen]
	    generated += sentence
	    print('----- Generating with seed: "' + sentence + '"')
	    sys.stdout.write(generated)

	    for i in range(800):
		x = np.zeros((1, maxlen, len(chars)))
		for t, char in enumerate(sentence):
		    x[0, t, char_indices[char]] = 1.

		preds = model.predict(x, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[next_index]

		generated += next_char
		sentence = sentence[1:] + next_char

		sys.stdout.write(next_char)
		sys.stdout.flush()
	    print()
	model.save_weights('./model_files/reports.rnn_char_weights.h5',overwrite=True)
	json_string = model.to_json()
	open('./model_files/reports.rnn_char_architecture.json', 'w').write(json_string)

