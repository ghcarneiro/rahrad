import cPickle as pkl
import numpy as np
import preprocess
from keras.models import Sequential, model_from_json, Graph
from keras.layers.core import Dense, TimeDistributedDense,Activation, Dropout,Lambda
import keras.datasets.data_utils
from math import ceil

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']

global rnn


def convert(item):
    numChars=ceil(len(item)/20)*20+20;
    lastPos=20-(numChars-len(item))
    if lastPos == 0:
	lastPos = 20
    X = np.zeros((1,len(item),72), dtype=np.bool)
    for t, char in enumerate(item):
        X[0,t, char_indices[char]] = 1
    #sentRep=[]
    for i in range(0,int(numChars),20):
	K = np.zeros((1,20,72),dtype=np.bool)
	#this line is trouble
	#K[0][i:i+20] = X[i:i+20]
	#sentRep += [rnn.predict(K,verbose=0)[0]]
	#sentRep = rnn.predict(K,verbose=0)[0][-1]
	#sentRep = rnn.predict(X,verbose=0)[0][lastPos-1]
	sentRep = rnn.predict(X,verbose=0)[0][-1]
    #reset rnn state
    rnn.reset_states()
    

    return sentRep 

def convertBatch(items):
    for item in enumerate(items):
	numChars=ceil(len(item)/20)*20+20;
	lastPos=20-(numChars-len(item))
	if lastPos == 0:
	    lastPos = 20
	X = np.zeros((numChars,72), dtype=np.bool)
	for t, char in enumerate(item):
	    X[t, char_indices[char]] = 1
	#sentRep=[]
	for i in range(0,int(numChars)-1,20):
	    K = np.zeros((1,20,72),dtype=np.bool)
	    K[0] = X[i:i+20]
	    #sentRep += [rnn.predict(K,verbose=0)[0]]
	    #sentRep = rnn.predict(K,verbose=0)[0][-1]
	    sentRep = rnn.predict(K,verbose=0)[0][lastPos-1]
	#reset rnn state
	rnn.reset_states()
    

    return sentRep 




rnn = model_from_json(open('./model_files/reports.rnn_char_architecture.json').read())
rnn.load_weights('./model_files/reports.rnn_char_weights.h5')
rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')
rnn.stateful = True


textR = preprocess.getReportsSectioned(REPORT_FILES)
fullText=""
for i in xrange(len(textR)):
    fullText += textR[i]
chars=set(fullText)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
fullText=""

trainAgain=True
if trainAgain == True:
	classify = Sequential()
	classify.add(Dense(len(textR),input_dim=512))
	classify.add(Activation('softmax'))
	classify.compile(loss='categorical_crossentropy',optimizer='rmsprop')
else:
	classify = model_from_json(open('./model_files/reports.classify_char_architecture.json').read())
	classify.load_weights('./model_files/reports.classify_char_weights.h5')


#process a batch then fit it... and so on on but in the same function
maxlen=20
step=3
batch=np.zeros((256,512),dtype=np.bool) 
labels=np.zeros((256,4),dtype=np.bool)
curBatchSize=0
times=0
print (convert("CT HEAD, X-RAYS OF UPPER SKULL REGION"))
print (convert("CT , LUNGS ARE DIFFLATED,POSSIBLITY OF"))

#train model
if trainAgain == True:
	for j in range(0,len(textR[i])/8-maxlen,step):
		for i in xrange(len(textR)):
		    sentence = textR[i][j:j+maxlen]
		    vector = convert(sentence) 	
		    batch[curBatchSize] = vector
		    label = i	    	
		    #check this covers all labels
		    for t in range(0,len(textR)):
			labels[curBatchSize][t] = label == t
		    
		    curBatchSize += 1
		    if curBatchSize == 256:
			classify.fit(batch, labels, batch_size=256, nb_epoch=1)
			times+=1
			curBatchSize = 0 
			print("Saving classifier")
			classify.save_weights('./model_files/reports.classify_char_weights.h5',overwrite=True)
			json_string = classify.to_json()
			open('./model_files/reports.classify_char_architecture.json', 'w').write(json_string)
			print("Finished Saving")

		    #if times == 7:
		#	break	
		#if times == 7:
		 #   break



#test model
preds = []
allPreds = []
batch=np.zeros((256,512),dtype=np.float) 
labels=np.zeros((256,4),dtype=np.bool)
for i in xrange(len(textR)):
	allPreds.append([[],[]])

for j in range(0,len(textR[i])/8-maxlen,step):
	for i in xrange(len(textR)):
	    sentence = textR[i][j:j+maxlen]
	    vector = convert(sentence) 
	    #this is the problem line	
	    for t in range(0,len(vector)):
		batch[curBatchSize,t] = vector[t]
	    label = i	    	
	    #check this covers all labels
	    for t in range(0,len(textR)):
		labels[curBatchSize][t] = label == t
	    
	    curBatchSize += 1
	    if curBatchSize == 256:
		preds += [classify.predict(batch,verbose=0)]
		print(preds)
		for i in xrange(len(preds)):
		    for k in xrange(len(textR)):	
			allPreds[k,0].append(preds[i][k])	
			allPreds[k,1].append(labels[i][k])
		times+=1
		curBatchSize = 0 
		print(allPreds)
		print(allPreds[0])
		print(allPreds[0][0])

	    if times == 7:
		break	
	if times == 7:
	    break

	    
