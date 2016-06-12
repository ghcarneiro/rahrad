import cPickle as pkl
import numpy as np
import preprocess
import random
from keras.models import Sequential, model_from_json, Graph
import keras.datasets.data_utils
from math import ceil

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']

global rnn
rnn = model_from_json(open('./model_files/reports.2rnn_char_architecture.json').read())
rnn.load_weights('./model_files/reports.2rnn_char_weights.h5')
rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')
rnn.stateful = True

# this is the class that the list is loaded with
class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.processed_sentence = ""
        self.diag_probs = []
        self.sent_probs = []
        self.diag_tag = ""
        self.sent_tag = ""
        self.report_id = ""
        self.feature_vector = []



print("Loading file")
textR = preprocess.getReportsSectioned(REPORT_FILES)
fullText=""
datas=[]
maxlen=20
step=3
print("Initialising tags and sentences")
#choosing 1000 random sentences evenly distributed
#between each type
random.seed(0)
minReportLen=len(textR[0])
for i in xrange(len(textR)):
    datas.append([])
    fullText += textR[i]
    if minReportLen > len(textR[i]):
	minReportLen = len(textR[i])

trainNow=True
if trainNow == True:
	for it in range(250):
	    i=random.randrange(0,minReportLen/8-maxlen,step)
	    for j in xrange(len(textR)):
		sentence=textR[j][i:i+maxlen] 
		for k in xrange(len(textR)):
		    sentRec = SentenceRecord(sentence)
		    sentRec.processed_sentence = sentence		
		    if j == k:
			sentRec.sent_tag="p"
		    else:
			sentRec.sent_tag="n"
		    sentRec.diag_tag=sentRec.sent_tag	
		    sentRec.report_id=j
		    
		    datas[k].append(sentRec)
else:
	for i in range(len(datas)):
	    data = open("tagged_data_corpus"+str(i)+".pkl","rb")
	    datas[i] = pkl.load(data)

		
'''	
for i in range(0,len(textR[0])/8-maxlen,step):
    for j in xrange(len(textR)):
	sentence = textR[j][i:i+maxlen]
	for k in xrange(len(textR)):
	    sentRec = SentenceRecord(sentence)
	    if i == k:
		sentRec.sent_tag="p"
	    else:
		sentRec.sent_tag="n"
	    data[k].append(sentRec)
'''

chars=set(fullText)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
fullText=""
testR=[]
'''
def convert(item):
    print('Preprocessing')
    X = np.zeros((1,20,len(chars)), dtype=np.bool)
    for t, char in enumerate(item):
        X[0,t+10, char_indices[char]] = 1
    sentRep = rnn.predict(X,verbose=0)[0]
    #reset rnn state
    rnn.reset_states()
    

    return sentRep 
'''
def convert(item):
    print('Preprocessing')
    totLen = len(item) / 20 * 20 + 20
    X = np.zeros((totLen,len(chars)), dtype=np.bool)
    for t, char in enumerate(item):
	X[t+totLen-len(item),char_indices[char]] = 1
    
    for k in range(0,len(item),20):
	X2 = np.zeros((1,20,len(chars)), dtype=np.bool)
	X2[0] = X[k:k+20]
	sentRep = rnn.predict(X2,verbose=0)[0]

    rnn.reset_states()
    return sentRep

# fill the feature_vector attribute of each item in the list according to the processed_sentence attribute
i=0
print("Calculating sentence Representations")
#maybe feature vector is same because item.sentence always 20 length
for data in datas:
    for item in data:
	print(item.sentence)
	vector = convert(item.sentence)
	item.feature_vector = vector
	print(len(vector))
    
    #print(item.feature_vector)
    print("Saving representation: "+str(i))
    pkl.dump(data, open("tagged_data_corpus_small"+str(i)+".pkl", "wb"))
    i+=1;
