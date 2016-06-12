from __future__ import print_function
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, TimeDistributedDense,Activation, Dropout,Lambda
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np
import random
import preprocess
import sys

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']
continueTraining =False
try:
    print('Loading Corpus')
    textR = preprocess.getReportsSectioned(REPORT_FILES)

    fullText=""
    maxlen = 20
    step=3
    totalTextLen = 5000000
    totalNumSent = totalTextLen/((maxlen/step)*20 )
    print('Formatting Sentences')
    sentences=[]
    labels=[]
    for i in xrange(len(textR)):
        for j in range(0,len(textR[i])/8-maxlen,step):
	    sentences.append(textR[i][j:j+maxlen])
	    labels.append(i); 
	fullText += textR[i]
    chars=set(fullText)
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
fullText=""

allSentences=[]
allLabels=[]

indice=0
for i in range(0,len(sentences)-totalNumSent,totalNumSent):
    allSentences.append(sentences[i:i+totalNumSent])
    allLabels.append(labels[i:i+totalNumSent])

rnn = model_from_json(open('./model_files/reports.rnn_char_architecture.json').read())
rnn.load_weights('./model_files/reports.rnn_char_weights.h5')
rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')

to_char = Sequential()
to_char.add(TimeDistributedDense(len(textR),input_shape=(512)))
to_char.add(Activation('softmax'))
to_char.compile(loss='categorical_crossentropy',optimizer='rmsprop')

for curSents in range(1,len(allSentences)):
        # cut the text in semi-redundant sequences of maxlen characters
        print('Preprocessing')
        X = np.zeros((totalNumSent, maxlen, 72), dtype=np.bool)
	z = np.zeros((totalNumSent,4),dtype=np.bool)
        #y = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
        for i, sentence in enumerate(allSentences[curSents]):
                for t, char in enumerate(sentence):
                        X[i, t, char_indices[char]] = 1
		for t in range(0,len(textR)-1):
			z[i, t] = allLabels[curSents][i]==t

	print("hi")

	
	sentRep = rnn.predict(X,verbose=0)
	print(sentRep)
	to_char.fit(sentRep, z, batch_size=256, nb_epoch=1)
    
