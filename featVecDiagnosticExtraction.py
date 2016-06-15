import cPickle as pkl
import numpy as np
import preprocess
from keras.models import Sequential, model_from_json, Graph
import keras.datasets.data_utils
from math import ceil

REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']

global rnn
rnn = model_from_json(open('./model_files/reports.2rnn_char_architecture.json').read())
rnn.load_weights('./model_files/reports.2rnn_char_weights.h5')
rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')
rnn.stateful = True


textR = preprocess.getReportsSectioned(REPORT_FILES)
fullText=""
for i in xrange(len(textR)):
    fullText += textR[i]
global chars
chars=set(fullText)
print(str(len(chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
fullText=""
testR=[]
'''
def convert(item):
    numChars=ceil(len(item)/20)*20+20;
    lastPos=20-(numChars-len(item))
    if lastPos == 0:
	lastPos = 20
    X = np.zeros((numChars,len(chars)), dtype=np.bool)
    for t, char in enumerate(item):
        X[t, char_indices[char]] = 1
    #sentRep=[]
    for i in range(0,int(numChars),20):
#	if lastPos == 20:
#		print(i)
#		print(len(item))
	K = np.zeros((1,20,len(chars)),dtype=np.bool)
	K[0] = X[i:i+20]
	#sentRep += [rnn.predict(K,verbose=0)[0]]
	#sentRep = rnn.predict(K,verbose=0)[0][-1]
	sentRep = rnn.predict(K,verbose=0)[0]
    #sentRep = rnn.predict(X,verbose=0)[0]
    #reset rnn state
    rnn.reset_states()
    

    return sentRep 
'''
'''
def convert(item):
    print('Preprocessing')
    X = np.zeros((1,len(item),len(chars)), dtype=np.bool)
    for t, char in enumerate(item):
        X[0,t, char_indices[char]] = 1
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


# tagged_data.pkl contains a list of 833 SentenceRecord objects
data2 = open("tagged_data.pkl","rb")
data2 = pkl.load(data2)

# fill the feature_vector attribute of each item in the list according to the processed_sentence attribute
for item in data2:
    vector = convert(item.sentence)
    item.feature_vector = vector
    #print(item.feature_vector)
    #print(item.sentence)

pkl.dump(data2, open("tagged_dataBetter.pkl", "wb"))

