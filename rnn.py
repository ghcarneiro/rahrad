from __future__ import division
from __future__ import print_function
import pickle
import csv
import re
from nltk import stem
import nltk
import gensim
import numpy as np
from scipy.spatial.distance import cdist
from numpy import newaxis
from sklearn import preprocessing
import preprocess
import keras
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
import time
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
    model = gensim.models.Word2Vec(reports, min_count=3, workers=4)
    model.init_sims(replace=True)
    model.save("./model_files/reports.word2vec_model")
    print("built word2vec")


def buildRNN():
    # Number of reports to process in each batch
    batchSize = 128
    # Detemined in the initial report processing
    maxLen = 0
    # Discard reports of length less than this value
    minLen = 10
    print("loading reports")
    reports = preprocess.getProcessedReports()
    reportsLen = len(reports)
    # Get max length of report and delete any reports with length less than 10 words
    for report in reports:
        length = len(report)
        if length > maxLen:
            maxLen = length
        if length < minLen:
            print(report)
            reports.remove(report)
    print("longest report length is: ", maxLen)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen, input_dim=100, return_sequences=True))
    m.add(LSTM(100, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    #Train the model over 10 epochs
    print("created LSTM model, training")
    for epoch in xrange(10):
        start=time.time()
        print("->epoch: ", epoch)
        for i in xrange(len(reports)):
            # Create batch in memory
            if ((i% batchSize) == 0):
                batch = np.zeros((batchSize,maxLen,100),dtype=np.float32)
            # Convert report to dense
            newReport = []
            for token in reports[i]:
                if token in word_model:
                    newReport.append(word_model[token])
            # Store report in batch
            batch[i%batchSize][0:len(newReport)][:]=np.asarray(newReport)
            # Train on batch
            if ((((i+1)% batchSize) == 0) or (i == (reportsLen-1))):
                print ("epoch: ",epoch,", ",i / reportsLen * 100)
                m.train_on_batch(batch,batch)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
    #Store the model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_architecture.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_weights.h5',overwrite=True)
    print("Trained model")

def buildPredictionsRNN():
    print("loading RNN model")
    model = model_from_json(open('./model_files/reports.rnn_architecture.json').read())
    model.load_weights('./model_files/reports.rnn_weights.h5')
    print("RNN model loaded")
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading reports")
    reports = preprocess.getProcessedReports()
    reportsLen = len(reports)
    print("loaded reports")
    print("generating predictions")
    predictions = np.zeros((len(reports),100))
    for i in xrange(reportsLen):
        # Create batch and pad individual reports
        newReport = []
        for token in reports[i]:
            if token in word_model:
                newReport.append(word_model[token])
        x=np.zeros((maxLen,100),dtype=np.float32)
        x[0:len(newReport)][:]=np.asarray(newReport)
        print(model.predict(x))
        predictions[i,:] = model.predict(x)[0]
        print(predictions[i,:])
        if ((i% 100) == 0):
            print (i / reportsLen * 100)
    file = open('./model_files/reports_rnn', 'w')
    pickle.dump(predictions, file)
    file.close()

def getSearchTerm(searchTerm):
    searchTerm = preprocess.textPreprocess(searchTerm)
    print("loading RNN model")
    model = model_from_json(open('./model_files/reports.rnn_architecture.json').read())
    model.load_weights('./model_files/reports.rnn_weights.h5')
    print("RNN model loaded")
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    for token in searchTerm:
        if token in word_model:
            newTerm.append(word_model[token])
    searchTerm = np.asarray(newTerm)
    searchTerm = m.predict(searchTerm)
    return searchTerm

def most_similar(searchTerm,topn=5):
    print("loading reports")
    reports = preprocess.getProcessedReports()
    print("loaded reports")
    print("loading report vectors")
    file = open('./model_files/reports_rnn', 'r')
    predictions = pickle.load(file)
    file.close()
    print("loaded report vectors")
    similarity = 1 - cdist(searchTerm,predictions,'cosine')
    idx = np.argsort(similarity)
    results = []
    for index in idx:
        if index < topn:
            result=[]
            result.append(reports[index])
            result.append(similarity[index])
            results.append(result)
        else:
            break
    return results

def buildSentenceRNN():
    # Number of reports to process in each batch
    batchSize = 128
    # Max number of words in sentence, detemined in the initial report processing
    maxLen = 0
    print("loading reports")
    reports = preprocess.getProcessedReports()
    # Get max length of sentence and prepare sentences for use
    sentences = []
    for report in reports:
        for sentence in report:
            length = len(sentence)
            if length > maxLen:
                maxLen = length
            sentences.append(sentence)
    sentenceLen = len(sentences)
    print("longest report length is: ", maxLen)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM sentence model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen, input_dim=100, return_sequences=True))
    m.add(LSTM(100, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    #Train the model over 10 epochs
    print("created LSTM sentence model, training")
    for epoch in xrange(10):
        start=time.time()
        print("->epoch: ", epoch)
        for i in xrange(len(sentences)):
            # Create batch in memory
            if ((i% batchSize) == 0):
                batch = np.zeros((batchSize,maxLen,100),dtype=np.float32)
            # Convert sentence to dense
            newSentence = []
            for token in sentence[i]:
                if token in word_model:
                    newSentence.append(word_model[token])
            # Store report in batch
            batch[i%batchSize][0:len(newReport)][:]=np.asarray(newSentence)
            # Train on batch
            if ((((i+1)% batchSize) == 0) or (i == (sentenceLen-1))):
                print ("epoch: ",epoch,", ",i / sentenceLen * 100)
                m.train_on_batch(batch,batch)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
    #Store the model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_sentence_architecture.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_sentence_weights.h5',overwrite=True)
    print("Trained sentence model")

def buildReportRNN():
    # Number of reports to process in each batch
    batchSize = 128
    # Max number of sentences in report, detemined in the initial report processing
    maxLen = 0
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("Loading sentence model")
    sentenceModel = model_from_json(open('./model_files/reports.rnn_sentence_architecture.json','r'))
    sentenceModel.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    print("Loaded sentence model")
    print("loading reports")
    reports = preprocess.getProcessedReports()
    reportsLen = len(reports)
    # Get max length of report and delete any reports with only 1 sentence
    for report in reports:
        length = len(report)
        if length > maxLen:
            maxLen = length
        if length == 1:
            print(report)
            reports.remove(report)
    print("longest report length is: ", maxLen)

    print('building LSTM model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen, input_dim=100, return_sequences=True))
    m.add(LSTM(100, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    #Train the model over 10 epochs
    print("created LSTM model, training")
    for epoch in xrange(10):
        start=time.time()
        print("->epoch: ", epoch)
        for i in xrange(len(reports)):
            # Create batch in memory
            if ((i% batchSize) == 0):
                batch = np.zeros((batchSize,maxLen,100),dtype=np.float32)
            # Convert report to dense
            newReport = []
            for token in reports[i]:
                if token in word_model:
                    newReport.append(word_model[token])
            # Store report in batch
            batch[i%batchSize][0:len(newReport)][:]=np.asarray(newReport)
            # Train on batch
            if ((((i+1)% batchSize) == 0) or (i == (reportsLen-1))):
                print ("epoch: ",epoch,", ",i / reportsLen * 100)
                m.train_on_batch(batch,batch)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
    #Store the model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_architecture.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_weights.h5',overwrite=True)
    print("Trained model")
