# :set softtabstop=4

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

# Load Saved Models?
continueTraining =False

# If you want to start from a certain epoch or section
startI=1
startIt=1

try:
    print('Loading Corpus')
    textR = preprocess.getReportsSectioned(REPORT_FILES)
    #text = open("./shakespeareEdit.txt").read().lower()

    fullText=""
    # Sentence length for training
    maxlen = 20
    # How far to move to start the next sentence
    step = 3
    reportSentences=[]
    print('Formatting Sentences')
    totalLength=0
    for i in xrange(len(textR)):
        sentences=[]
        for j in range(0,len(textR[i])/8-maxlen,step):
	    # [ sentence, next character , scan type label ] 
            sentences.append([textR[i][j:j+maxlen],textR[i][j+maxlen],i])
	totalLength+= len(sentences)
	fullText += textR[i]
        reportSentences += sentences
    print("Number of Sentences")
    print(totalLength)
    # Get all possible characters in dataset
    chars=set(fullText)
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

# Set up character transformation arrays
print('total chars:', len(chars))
# Character to one hot
char_indices = dict((c, i) for i, c in enumerate(chars))
# One hot to Character 
indices_char = dict((i, c) for i, c in enumerate(chars))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network(input_dim):
    ''' Double layered RNN which returns the 512 hidden layer representation of the last uni
    '''
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    # return_sequences=False , only return one 512 representation for the entire sentence
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    return model 

def euclidean_distance(inputs):
    assert len(inputs) == 2, ('Euclidean distance needs '
                              '2 inputs, %d given' % len(inputs))
    u, v = inputs.values()
    eps=1e-9
    return K.sqrt(K.maximum(K.sum(K.square(u - v), axis=1, keepdims=True),eps)) 
	

print('Build model...')
if not continueTraining:
    #RNN w/ Siamese
    base_network = create_base_network((maxlen,len(chars),))
    #RNN w/o Siamese
    norm_network = create_base_network((maxlen,len(chars),))
    # Transforms a 512 representation to a one hot character rep
    # Decoding Section - which is a Neural Network
    to_char = Sequential()
    to_char.add(Dense(len(chars),input_shape=(512,)))
    to_char.add(Activation('softmax'))
    # This is for RNN W/O SIAMESE
    to_char2 = Sequential()
    to_char2.add(Dense(len(chars),input_shape=(512,)))
    to_char2.add(Activation('softmax'))
else:
    # Base Network - Encoding RNN - No Decoding    
    base_network = model_from_json(open('./model_files/reports.2rnn_char_architecture.json').read())
    base_network.load_weights('./model_files/reports.2rnn_char_weights.h5')
    norm_network = model_from_json(open('./model_files/reports.2rnn_norm_char_architecture.json').read())
    norm_network.load_weights('./model_files/reports.2rnn_norm_char_weights.h5')
    to_char = model_from_json(open('./model_files/reports.2rep2Char_char_architecture.json').read())
    to_char.load_weights('./model_files/reports.2rep2Char_char_weights.h5')
    to_char2 = model_from_json(open('./model_files/reports.2rep2Char2_char_architecture.json').read())
    to_char2.load_weights('./model_files/reports.2rep2Char2_char_weights.h5')


# SIAMESE GRAPH
g = Graph()
g.add_input(name='input_a', input_shape=(maxlen,len(chars)))
g.add_input(name='input_b', input_shape=(maxlen,len(chars)))
# Node for joining outputs of RNN Representation for each sentence.
g.add_shared_node(base_network, name='shared', inputs=['input_a', 'input_b'],
                  merge_mode='join')
# Get the difference between the two RNN's representations.
g.add_node(Lambda(euclidean_distance,output_shape=eucl_dist_output_shape), name='d', input='shared')
g.add_output(name='output', input='d')
g.compile(loss={'output': contrastive_loss}, optimizer='rmsprop')

# CHARACTER BY CHARACTER TRAINING OF RNN
g2 = Graph()
g2.add_input(name='input',input_shape=(maxlen,len(chars)))
g2.add_node(base_network,name='rnn',input='input')
g2.add_node(to_char,name='charRep',input='rnn')
g2.add_output(name='output',input='charRep')
g2.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop')

# RNN W/O SIAMESE
g3 = Graph()
g3.add_input(name='input',input_shape=(maxlen,len(chars)))
g3.add_node(norm_network,name='rnn',input='input')
g3.add_node(to_char2,name='charRep',input='rnn')
g3.add_output(name='output',input='charRep')
g3.compile(loss={'output': 'categorical_crossentropy'},optimizer='rmsprop')

print('Randomizing pairing of sentences')
random.seed(0)
random.shuffle(reportSentences)
print('Splitting data into Sections')
indice=0

# This is due to memory constrains - split up the data into sections
totalTextLen = 5000000
# make sure this is even
totalNumSent = totalTextLen/((maxlen/step)*20 )
# For Debugging
# totalNumSent = 256
for i in range(0,len(reportSentences)-totalNumSent,totalNumSent):
    allSentences.append(reportSentences[i:i+totalNumSent])	
    for j in range(0,len(allSentences[indice])-1,2):
	# If from the same scan type - 1 else 0 - used for RNN loss.	
	allSentences[indice][j][2] = allSentences[indice][j][2] == allSentences[indice][j+1][2]
	allSentences[indice][j+1][2] = allSentences[indice][j][2] 
    indice+= 1

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

for epoch in range(startI, 60):
    for curSents in range(startIt,len(allSentences)):
	startIt=1
	# cut the text in semi-redundant sequences of maxlen characters
	print('Preprocessing')
	X = np.zeros((totalNumSent, maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((totalNumSent, len(chars)), dtype=np.bool)
	z = np.zeros((totalNumSent), dtype=np.bool)
	for i, sentence in enumerate(allSentences[curSents]):
		for t, char in enumerate(sentence[0]):
			X[i, t, char_indices[char]] = 1
		if i % 2 == 0:
		    z[i] = sentence[2] 
                    z[i+1] = sentence[2] 
		y[i,char_indices[sentence[1]]] = 1
		
	print('Finished Preprocessing')

	print()
	print('-' * 50)
	print('Epoch', epoch)
	print('Section',curSents)
        # Split paired sentences to their corresponding side of the Siamese Network
	X1 = X[::2][:]
	X2 = X[1::2][:]
	z2 = z[::2][:]

	# Train Siamese
        g.fit({'input_a': X1, 'input_b': X2, 'output': z2}, batch_size=256,nb_epoch=1)
	# Train RNN Section
	g2.fit({'input':X,'output':y},batch_size=256,nb_epoch=1)
	# Train RNN W/O Siamese
	g3.fit({'input':X,'output':y},batch_size=256,nb_epoch=1)

	# Save all models 
	base_network.save_weights('./model_files/reports.2rnn_char_weights.h5',overwrite=True)
	json_string = base_network.to_json()
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
	    # Choose a random sentence to start from for sentence generation 
	    random_sentence = random.randint(0, totalNumSent - 1)

		# Generate text for different sampling diversity's 
		for diversity in [0.2, 0.5, 1.0, 1.2]:
		    print()
		    print('----- diversity:', diversity)
		    generated = ''
		    # Get the random sentence
		    sentence = allSentences[curSents][random_sentence][0] 
		    generated += sentence
		    # Begin Generating 800 characters worth
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
	    
