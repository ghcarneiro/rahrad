'''mple script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, TimeDistributedDense,Activation, Dropout,Lambda
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np
import random
import preprocess
import sys

allSentences=[]

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return (K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))))

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']
continueTraining =True
try:
    print('Loading Corpus')
    # change textR so that each files reports is one big string
    textR = preprocess.getReportsSectioned(REPORT_FILES)
    #text = open("./shakespeareEdit.txt").read().lower()
    fullText=""
    maxlen = 20
    totalTextLen = 5000000
    step = 3
    # make sure this is even
    totalNumSent = totalTextLen/((maxlen/step)*20 )
    #totalNumSent=833332
    #totalNumSent=20832
    print(totalNumSent)
    #totalNumSent = 256 
    reportSentences=[]
    print('Formatting Sentences')
    print(len(textR))
    totalLength=0
    for i in xrange(len(textR)):
        sentences=[]
        for j in range(0,len(textR[i])/8-maxlen,step):
            sentences.append([textR[i][j:j+maxlen],textR[i][j+maxlen],i])
	totalLength+= len(sentences)
	fullText += textR[i]
        reportSentences += sentences
    print(totalLength)
    chars=set(fullText)
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()
'''
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
    print(len(textR))
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
'''

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    return model 

def euclidean_distance(inputs):
    assert len(inputs) == 2, ('Euclidean distance needs '
                              '2 inputs, %d given' % len(inputs))
    u, v = inputs.values()
    #c = np.matrix(u)
    #d = np.matrix(v)
    #return np.mean(np.linalg.norm(c-d,axis=1))
    #TMP=K.mean(K.sum(K.sum(K.square(u-v),axis=2),axis=1))
    #return TMP
    eps=1e-9
    #return K.sqrt(K.sum(K.sum(K.square(u - v), axis=2, keepdims=True),axis=1))
    # recheck off website what the original was
    #return K.sqrt(K.maximum(K.sum(K.sum(K.square(u - v), axis=2, keepdims=True),axis=1),eps))
    return K.sqrt(K.maximum(K.sum(K.square(u - v), axis=1, keepdims=True),eps)) 
	

print('Build model...')
if not continueTraining:
    base_network = create_base_network((maxlen,len(chars),))
    norm_network = create_base_network((maxlen,len(chars),))
    to_char = Sequential()
    to_char.add(Dense(len(chars),input_shape=(512,)))
    to_char.add(Activation('softmax'))
    to_char2 = Sequential()
    to_char2.add(Dense(len(chars),input_shape=(512,)))
    to_char2.add(Activation('softmax'))



else:
    base_network = model_from_json(open('./model_files/reports.2rnn_char_architecture.json').read())
    base_network.load_weights('./model_files/reports.2rnn_char_weights.h5')
    norm_network = model_from_json(open('./model_files/reports.2rnn_norm_char_architecture.json').read())
    norm_network.load_weights('./model_files/reports.2rnn_norm_char_weights.h5')
    to_char = model_from_json(open('./model_files/reports.2rep2Char_char_architecture.json').read())
    to_char.load_weights('./model_files/reports.2rep2Char_char_weights.h5')
    to_char2 = model_from_json(open('./model_files/reports.2rep2Char2_char_architecture.json').read())
    to_char2.load_weights('./model_files/reports.2rep2Char2_char_weights.h5')


g = Graph()
g.add_input(name='input_a', input_shape=(maxlen,len(chars)))
g.add_input(name='input_b', input_shape=(maxlen,len(chars)))
g.add_shared_node(base_network, name='shared', inputs=['input_a', 'input_b'],
                  merge_mode='join')
g.add_node(Lambda(euclidean_distance,output_shape=eucl_dist_output_shape), name='d', input='shared')
g.add_output(name='output', input='d')

g.compile(loss={'output': contrastive_loss}, optimizer='rmsprop')

g2 = Graph()
g2.add_input(name='input',input_shape=(maxlen,len(chars)))
g2.add_node(base_network,name='rnn',input='input')
g2.add_node(to_char,name='charRep',input='rnn')
g2.add_output(name='output',input='charRep')
g2.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop')

g3 = Graph()
g3.add_input(name='input',input_shape=(maxlen,len(chars)))
g3.add_node(norm_network,name='rnn',input='input')
g3.add_node(to_char2,name='charRep',input='rnn')
g3.add_output(name='output',input='charRep')
g3.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop')


model = base_network

print('Randomizing')
random.seed(0)
random.shuffle(reportSentences)
print('Segmenting what we will use at a time')
indice=0
for i in range(0,len(reportSentences)-totalNumSent,totalNumSent):
    allSentences.append(reportSentences[i:i+totalNumSent])	
    for j in range(0,len(allSentences[indice])-1,2):
	allSentences[indice][j][2] = allSentences[indice][j][2] == allSentences[indice][j+1][2]
	allSentences[indice][j+1][2] = allSentences[indice][j][2] 
    indice+= 1

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
startI=7
startIt=95
for iteration in range(startI, 60):
    for curSents in range(startIt,len(allSentences)):
	startIt=1
	# cut the text in semi-redundant sequences of maxlen characters
	print('Preprocessing')
	X = np.zeros((totalNumSent, maxlen, len(chars)), dtype=np.bool)
	#y = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
	y = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
	z = np.zeros((totalNumSent), dtype=np.bool)
	for i, sentence in enumerate(allSentences[curSents]):
		for t, char in enumerate(sentence[0]):
			X[i, t, char_indices[char]] = 1
		if i % 2 == 0:
		 #   y[i, 0:len(chars)] = [sentence[2]]*(len(chars))
		    z[i] = sentence[2] 
                    z[i+1] = sentence[2] 
		    #y[i, 0:len(chars)] = sentence[2]
		#else:
		#y[i, char_indices[sentence[1]]] = 1
		y[i,char_indices[sentence[1]]] = 1
		
	print('Finished Preprocessing')
	# train the model, output generated text after each iteration
	print()
	print('-' * 50)
	print('Iteration', iteration)
        # base network takes the both sides of input
        # model takes pairs of inputs
	X1 = X[::2][:]
	print(X.shape)
	print(X1.shape)
	X2 = X[1::2][:]
	z2 = z[::2][:]
	#z2 = z2.reshape(z2.shape[0],1)
	print(z2.shape)
	print(curSents)
	if curSents == 135:
		print(z)
	print(z)
        g.fit({'input_a': X1, 'input_b': X2, 'output': z2}, batch_size=256,nb_epoch=1)
	g2.fit({'input':X,'output':y},batch_size=256,nb_epoch=1)
	g3.fit({'input':X,'output':y},batch_size=256,nb_epoch=1)

	random_sentence = random.randint(0, totalNumSent - 1)
	model.save_weights('./model_files/reports.2rnn_char_weights.h5',overwrite=True)
	json_string = model.to_json()
	open('./model_files/reports.2rnn_char_architecture.json', 'w').write(json_string)
	to_char.save_weights('./model_files/reports.2rep2Char_char_weights.h5',overwrite=True)
	json_string = to_char.to_json()
	open('./model_files/reports.2rep2Char_char_architecture.json', 'w').write(json_string)
	to_char2.save_weights('./model_files/reports.2rep2Char2_char_weights.h5',overwrite=True)
	json_string = to_char2.to_json()
	open('./model_files/reports.2rep2Char2_char_architecture.json', 'w').write(json_string)
	norm_network.save_weights('./model_files/reports.2rnn_norm_char_weights.h5',overwrite=True)
	json_string = norm_network.to_json()
	open('./model_files/reports.2rnn_norm_char_architecture.json', 'w').write(json_string)
	if (curSents-1) % 10 == 0:
		for diversity in [0.2, 0.5, 1.0, 1.2]:
		    print()
		    print('----- diversity:', diversity)
		    generated = ''
		    sentence = allSentences[curSents][random_sentence][0] 
		    generated += sentence
		    for i in range(800):
			x = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
			   x[0, t, char_indices[char]] = 1.
	     
			preds = g2.predict({'input':x}, verbose=0)
			preds = preds['output'][0]
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]
			generated += next_char
			sentence = sentence[1:] + next_char
			sys.stdout.write(next_char)
			sys.stdout.flush()
		    print()
	    
