
from __future__ import print_function
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
import preprocess
import sys

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']

try: 
    textR = preprocess.getReports(REPORT_FILES)
    text = ""
    #text = open("./shakespeareEdit.txt").read().lower()
    for i in xrange(len(textR)):
	text += textR[i]
	
    #text=text[1:5000000]
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))



maxlen = 20
'''
model = model_from_json(open('./model_files/reports.rnn_norm_char_architecture.json').read())
model.load_weights('./model_files/reports.rnn_norm_char_weights.h5')
to_char = model_from_json(open('./model_files/reports.rep2Char2_char_architecture.json').read())
to_char.load_weights('./model_files/reports.rep2Char2_char_weights.h5')
'''
model = model_from_json(open('./model_files/reports.2rnn_char_architecture.json').read())
model.load_weights('./model_files/reports.2rnn_char_weights.h5')
to_char = model_from_json(open('./model_files/reports.2rep2Char_char_architecture.json').read())
to_char.load_weights('./model_files/reports.2rep2Char_char_weights.h5')
model.stateful = True



g2 = Graph()
g2.add_input(name='input',input_shape=(maxlen,len(chars)))
g2.add_node(model,name='rnn',input='input')
g2.add_node(to_char,name='charRep',input='rnn')
g2.add_output(name='output',input='charRep')
g2.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop')


#for diversity in [0.2, 0.5, 1.0, 1.2]:
for diversity in range(100):
	#print()
	#print('----- diversity:', diversity)

	generated = ''
	sentence = '' 
#	for i in range(1,maxlen):
#		sentence += ' '
	#sentence += 'THE MEDICAL PRACTION'
	sentence +=  '                  CT'
	#sentence = input("What's your sentence? ")
	
	for char in sentence:
		sys.stdout.write(char)
	#sentence +=  '               CANCER';
	generated += sentence
	#sys.stdout.write(generated)
	
	for i in range(1000):
		x = np.zeros((1,maxlen,len(chars) ))
		for t, char in enumerate(sentence):
			x[0, t, char_indices[char]] = 1.
	
		preds = g2.predict({'input':x}, verbose=0)
		preds = preds['output'][0]

		#preds = model.predict(x, verbose=0)
		#preds = preds[0];
		# this change is so we only get the last character for a return sequence=true model
		#preds = preds[-1]
		#next_index = sample(preds, diversity)
		next_index = sample(preds, 0.5)
		next_char = indices_char[next_index]
	
		generated += next_char
		sentence = sentence[1:] + next_char
	
		sys.stdout.write(next_char)
		sys.stdout.flush()
	print()
	print()
	print()
	print()


