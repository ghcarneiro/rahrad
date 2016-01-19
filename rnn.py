from __future__ import division
from __future__ import print_function
import pickle
import csv
import re
from nltk import stem
import nltk
import gensim
import keras
import numpy as np
from sklearn import preprocessing
import preprocess
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
REPORT_FILES = ['nlp_data/CleanedBrainsFull.csv','nlp_data/CleanedCTPAFull.csv','nlp_data/CleanedPlainabFull.csv','nlp_data/CleanedPvabFull.csv']
REPORT_FILES_BRAINS = ['nlp_data/CleanedBrainsFull.csv']
REPORT_FILES_CTPA = ['nlp_data/CleanedCTPAFull.csv']
REPORT_FILES_PLAINAB = ['nlp_data/CleanedPlainabFull.csv']
REPORT_FILES_PVAB = ['nlp_data/CleanedPvabFull.csv']

REPORT_FILES_LABELLED = ['nlp_data/CleanedBrainsLabelled.csv','nlp_data/CleanedCTPALabelled.csv','nlp_data/CleanedPlainabLabelled.csv','nlp_data/CleanedPvabLabelled.csv']
REPORT_FILES_LABELLED_BRAINS = ['nlp_data/CleanedBrainsLabelled.csv']
REPORT_FILES_LABELLED_CTPA = ['nlp_data/CleanedCTPALabelled.csv']
REPORT_FILES_LABELLED_PLAINAB = ['nlp_data/CleanedPlainabLabelled.csv']
REPORT_FILES_LABELLED_PVAB = ['nlp_data/CleanedPvabLabelled.csv']

DIAGNOSES = ['Brains','CTPA','Plainab','Pvab']
# global variables, loaded during first call to text preprocessing
# specialist dictionary
medical = dict()

# runs the preprocessing procedure to the supplied text
# input is string of text to be processed
# output is the same string processed
# set minimal to true for minimal text preprocessing
def textPreprocess(text):
    #load dictionary of specialist lexicon
    global medical
    if not medical:
        file = open('./dictionary_files/medical.pkl', 'r')
        medical = pickle.load(file)
        file.close()

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    report = []
    sentences = sentence_token.tokenize(text.strip())

    # for sentence in sentences:
    sentences = sentences.lower() # convert to lower-case
    # Split on non alphanumeric and non hyphen characters and keep delimiter
    sentences = re.split("([^\w\-]+)||\b", sentences)
    # Delete whitespace tokens
    sentences = [word.replace(' ','') for word in sentences]
    sentences = filter(None, sentences)
    text = []
    #look up variable length sequences of words in medical dictionary, stem them if not present
    for sentence in sentences:
        numTokens = 5 #phrases up to 5 words long
        while numTokens > 0:
        	processedText=[]
        	start=0
        	#Check each phrase of n tokens while there are sufficient tokens after
        	while start < (len(sentence) - numTokens):
        		phrase=sentence[start]
        		nextToken=1
        		while nextToken < numTokens:
        			#add the next tokens to the current one
        			phrase = phrase+" "+sentence[start+nextToken]
        			nextToken += 1
        		if phrase in medical:
        			#convert tokens to one token from specialist
        			processedText.append(medical[phrase])
        			# skip the next tokens
        			start += (numTokens)
        		elif numTokens == 1:
        			#individual tokens, stem them if not in specialist and keep
        			processedText.append(stem.snowball.EnglishStemmer().stem(phrase))
        			start += 1
        		else:
        			#token not part of phrase, keep
        			processedText.append(sentence[start])
        			start += 1
        	#Keep remaining tokens without enough tokens after them
        	while start < len(sentence):
        		processedText.append(sentence[start])
        		start += 1
        	sentence = processedText
        	numTokens -= 1
        text.append(sentence)
    text.append("end_rep")

    return(text)

# fetches the raw reports, preprocesses them, then saves them as report files
# input must be an array of fileNames to preprocess. By default, preprocesses all reports in directory.
def preprocessReports(fileNames=REPORT_FILES):
	for j in range(len(fileNames)):

		reports = preprocess.getReports([fileNames[j]])
		#reports = getSentences([fileNames[j]])
		print("loading finished")


		for i in xrange(len(reports)):
			print (i / len(reports) * 100)
			reports[i] = textPreprocess(reports[i])
		print("preprocessing finished")

		file = open('./model_files/reports_list_' + DIAGNOSES[j], 'w')
		pickle.dump(reports, file)
		file.close()

		print("report saved")

# Loads all reports and converts words to integer ids
# Stores the new reports to reports_list_all_ints with first field containing word integer range
# Deletes any words that appear less than 5 times in the entirety of the reports
def reportToInts():
    # load all the reports
    reports = preprocess.getProcessedReports()
    print("files loaded")
    # build dictionary
    dictionary = gensim.corpora.Dictionary(reports)
    dictionary.filter_extremes(no_below=5)
    dictionary.compactify()
    print("dictionary created")
    # Map the words to their integer ids
    mapping = dict()
    for item in dictionary.items():
        mapping[item[1]] = item[0]
    # Convert all the words in the reports to their ids
    for report in reports:
        newReport=[]
        for token in report:
            if token in mapping:
                newReport.append(int(mapping[token]))
        report = newReport
    # Store the word id range to the start of the reports list
    reports = [len(dictionary.items())] + reports
    print("Reports converted to ints")
    # Store the new reports to a file
    file = open('./model_files/reports_list_all_ints', 'w')
    pickle.dump(reports, file)
    file.close()
    print("Done")

def buildWord2Vec():
    print("building word2vec model")
    reports = preprocess.getProcessedReports()
    model = gensim.models.Word2Vec(reports, min_count=5, workers=4)
    model.init_sims(replace=True)
    model.save("./model_files/reports.word2vec_model")
    print("built word2vec")


def buildRNN():
    batch=[]
    batchLen = 0
    maxLen = 0
    longestReport =[]
    print("loading reports")
    reports = preprocess.getProcessedReports()
    reportsLen = len(reports)
    #Get max length of report
    for report in reports:
        length = len(report)
        if length > maxLen:
            maxLen = length
            longestReport = report
    print("longest report is: ", maxLen)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen, input_dim=100, return_sequences=True))
    m.add(LSTM(100, return_sequences=True))
    # m.add(Activation('linear'))
    m.compile(loss='mse', optimizer='adam')
    print("created LSTM model, training")
    for epoch in xrange(10):
        print("epoch: ", epoch)
        for i in xrange(len(reports)):
            # Create batch and pad individual reports
            newReport = []
            for token in report[i]:
                if token in word_model:
                    newReport.append(word_model[token])
            batch.append(pad_sequences(newReport, maxlen=maxLen))
            # Train on batches of size 32
            if ((i+1) % 32 == 0):
                print (i / reportsLen * 100)
                x = np.array(batch)
                m.train_on_batch(x,x)
                batch=[]
    m.save_weights('./model_files/rnn.h5')
