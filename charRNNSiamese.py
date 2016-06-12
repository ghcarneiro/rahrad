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
from keras import backend as K
import numpy as np
import random
import preprocess
import sys

allSentences=[]

def siamese(y_true,y_pred):
    a=y_pred[0::2]
    b=y_pred[1::2]
    diff=K.sum((a-b)**2,axis=1)
    y_true=y_true[0::2]
    return K.mean((diff-y_true)**2)

def siamese(y_true,y_pred):
    a=y_pred[0::2]
    b=y_pred[1::2]
    exp=y_true[1::2]
    diff=((a-b)**2).sum(axis=1,keepdims=True)
    #diff = K.sum((a-b)**2,axis=1)
    diffR=((a-exp)**2).sum(axis=1,keepdims=True)
    #diffR=K.sum((a-exp)**2,axis=1)
    y_true=y_true[0::2]
    #return K.mean((diff+diffR-y_true)**2,axis=-1)
    return((diff+diffR-y_true)**2).mean()
    #return((diff-y_true)**2).mean()

def categoricalcrossentropy(y_true,y_pred):
    loss = K.categorical_crossentropy(y_pred,y_true)
    print(y_true)
    print(y_pred)
    print(loss)
    return loss

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']
continueTraining = False 

try: 
    print('Loading Corpus')
    textR = preprocess.getReports(REPORT_FILES)
    #text = open("./shakespeareEdit.txt").read().lower()
    fullText=""
    maxlen = 20 
    totalTextLen = 5000000
    step = 3
    # make sure this is even
    totalNumSent = totalTextLen/((maxlen/step) * 2)
    #totalNumSent = 256 
    reportSentences=[]
    print('Formatting Sentences')
    for i in xrange(len(textR)):
	sentences=[]
	for j in range(0,len(textR[i])/8-maxlen,step):
	    sentences.append([textR[i][j:j+maxlen],textR[i][j+maxlen],i])
	reportSentences += sentences
	fullText += textR[i]
    chars=set(fullText)
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(fullText)/2)
fullText=""

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
    #model.compile(loss=categoricalcrossentropy, optimizer='rmsprop')
    #model.compile(loss=siamese, optimizer='rmsprop')
else:
    m = model_from_json(open('./model_files/reports.rnn_char_architecture.json').read())
    m.load_weights('./model_files/reports.rnn_char_weights.h5')

print('Randomizing')
random.shuffle(reportSentences)
print('Segmenting what we will use at a time')
indice=0
for i in range(0,len(reportSentences)-totalNumSent,totalNumSent):
    allSentences.append(reportSentences[i:i+totalNumSent])	
    for j in range(0,len(allSentences[indice])-1,2):
	allSentences[indice][j][2] = allSentences[indice][j][2] == allSentences[indice][j+1][2]
	allSentences[indice][j+1][2] = allSentences[indice][j][2] == allSentences[indice][j+1][2]
    indice+= 1

print('Deleting unneeded memory')
#del(reportSentences)
#not required in memory any longer


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


for iteration in range(1, 60):
    for curSents in range(1,len(allSentences)):
	# cut the text in semi-redundant sequences of maxlen characters
	print('Preprocessing')
	X = np.zeros((totalNumSent, maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
	#z = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
	for i, sentence in enumerate(allSentences[curSents]):
		for t, char in enumerate(sentence[0]):
			X[i, t, char_indices[char]] = 1
		#if i % 2 == 0:
		 #   y[i, 0:len(chars)] = [sentence[2]]*(len(chars))
		  #  y[i+1,char_indices[sentence[1]]] = 1
		    #y[i, 0:len(chars)] = sentence[2]
		#else:
		y[i, char_indices[sentence[1]]] = 1
	print('Finished Preprocessing')
	# train the model, output generated text after each iteration
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=256, nb_epoch=1)
	random_sentence = random.randint(0, totalNumSent - 1)
	model.save_weights('./model_files/reports.rnn_char_weights.h5',overwrite=True)
	json_string = model.to_json()
	open('./model_files/reports.rnn_char_architecture.json', 'w').write(json_string)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
	    print()
	    print('----- diversity:', diversity)
	    generated = ''
	    sentence = allSentences[curSents][random_sentence][0] 
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
    
