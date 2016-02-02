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
# from numpy import newaxis
# from sklearn import preprocessing
import preprocess
import keras
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers.core import Dense, Dropout, Activation
# from keras.layers.embeddings import Embedding
import theano
import math
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
# output is the same string processed and tokenised into words and sentences
def textPreprocess(text):
    #load dictionary of specialist lexicon
    global medical
    if not medical:
        file = open('./dictionary_files/medical.pkl', 'r')
        medical = pickle.load(file)
        file.close()

    # Covert to lower case
    text = text.lower()
    # Split text into sentences
    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_token.tokenize(text.strip())

    text = []

    for sentence in sentences:
        # Split on non alphanumeric and non hyphen characters and keep delimiter
        sentence = re.split("([^\w\-]+)||\b", sentence)
        # Delete whitespace tokens
        sentence = [word.replace(' ','') for word in sentence]
        sentence = filter(None, sentence)
        #look up variable length sequences of words in medical dictionary, stem them if not present
        numTokens = 5 #phrases up to 5 words long
        while (numTokens > 0):
            processedText=[]
            start=0
            #Check each phrase of n tokens while there are sufficient tokens after
            while (start <= (len(sentence) - numTokens)):
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
            while (start < len(sentence)):
                processedText.append(sentence[start])
                start += 1
            sentence = processedText
            numTokens -= 1
        text.append(sentence)
    text.append(["end_rep"])

    return(text)

# fetches the raw reports, preprocesses them, then saves them to a single reports file
# also fetches all the sentences from these reports and saves them to a single sentences file
# input must be an array of fileNames to preprocess. By default, preprocesses all reports in directory.
def preprocessReports(fileNames=REPORT_FILES):
    allReports = []
    allSentences = []
    for j in range(len(fileNames)):
    	reports = preprocess.getReports([fileNames[j]])
    	print("loading finished")
    	for i in xrange(len(reports)):
            print (i / len(reports) * 100)
            reports[i] = textPreprocess(reports[i])
            allSentences = allSentences + reports[i]
    	print("preprocessing finished")
        allReports = allReports + reports

    file = open('./model_files/reports_full', 'w')
    pickle.dump(allReports, file)
    file.close()
    print("reports saved")

    file = open('./model_files/reports_sentences_full', 'w')
    pickle.dump(allSentences, file)
    file.close()
    print("sentences saved")

# retrieves all reports that have been preprocessed
# output is an array containing the processed reports
def getProcessedReports():
	file = open('./model_files/reports_full', 'r')
	reports = pickle.load(file)
	file.close()

	return reports

# retrieves all sentences that have been preprocessed
# output is an array containing the processed reports
def getProcessedSentences():
	file = open('./model_files/reports_sentences_full', 'r')
	sentences = pickle.load(file)
	file.close()

	return sentences

# Builds a word2vec model on the processed sentences extracted from reports
# This function is required to create the dense word embeddings
def buildWord2VecSentences():
    print("loading sentences")
    sentences = getProcessedSentences()
    print("loaded sentences")
    print("building word2vec model")
    model = gensim.models.Word2Vec(sentences, min_count=3, workers=4)
    model.init_sims(replace=True)
    model.save("./model_files/reports.word2vec_model")
    print("built word2vec")

# function to test the functionality of Word2Vec
def testWord2VecModel():
    print("loading word2vec model")
    model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print(model)
    # model = gensim.models.Word2Vec.load("zzmodel")
    print("----------------------------------similarity test")
    print(model.similarity("head","brain"))
    print("----------------------------------raw numpy vector of word")
    print(model["age"])
    print("----------------------------------word to vector and back")
    print(model.most_similar(positive=[model["hemorrhage"]],topn=1)[0][0])
    print("----------------------------------remove outlier")
    print(model.doesnt_match("hours four age".split()))
    print("----------------------------------similar words")
    print(model.most_similar("hemorrhage"))

    print("script finished")

# Build an RNN on sentences using a fixed maximum sentence length and batches
def buildSentenceRNN():
    # Number of sentences to process in each batch
    batchSize = 128
    # Max number of words in sentence, detemined in the initial report processing
    maxLen = 50
    print("loading sentences")
    sentences = getProcessedSentences()
    numSentences = len(sentences)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen-1, input_dim=100, return_sequences=True))
    m.add(LSTM(100, return_sequences=True))
    m.compile(loss='mse', optimizer='adam')
    temp = np.zeros((maxLen,100),dtype=np.float32)
    #Train the model over 10 epochs
    print("created LSTM sentence model, training")
    for epoch in xrange(10):
        start=time.time()
        print("->epoch: ", epoch)
        for i in xrange(numSentences):
            # Create batch in memory
            if ((i% batchSize) == 0):
                batch = np.zeros((batchSize,maxLen-1,100),dtype=np.float32)
                expected = np.zeros((batchSize,maxLen-1,100),dtype=np.float32)
            # Convert sentence to dense
            newSentence = []
            for token in sentences[i]:
                if token in word_model:
                    newSentence.append(word_model[token])
            # Trim sentences greater than maxLen
            newSentence = newSentence[:maxLen-1]
            # Store sentence in batch
            if len(newSentence) > 0:
                temp = np.asarray(newSentence)
                # Last word of sentence is not used as input
                batch[i%batchSize][0:len(newSentence)-1][:]=temp[0:len(newSentence)-1][:]
                # First word of sentence is not used as expected output (right shifted input)
                expected[i%batchSize][0:len(newSentence)-1][:]=temp[1:len(newSentence)][:]
            # Train on batch
            if ((((i+1)% batchSize) == 0) or (i == (numSentences-1))):
                print ("epoch: ",epoch,", ",i / numSentences * 100)
                m.train_on_batch(batch,batch)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
    #Store the model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_sentence_architecture.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_sentence_weights.h5',overwrite=True)
    print("Trained sentence model")

# Build a RNN on sentences using a fixed window size and no batches
# Due to the updating of weights in keras this function is unusable due to computation time required
def buildSentenceRNNWindowed():
    # Max number of words in training set, detemined in the initial report processing
    timeWindow = 10
    timeSteps = timeWindow + 1
    trimmed = 0
    print("loading sentences")
    sentences = getProcessedSentences()
    numSentences = len(sentences)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM model...')
    model = Sequential()
    model.add(LSTM(100, input_length=timeSteps, input_dim=100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.compile(loss='mse', optimizer='adam')
    get_activations = theano.function([model.layers[0].input], model.layers[0].get_output(train=False), allow_input_downcast=True)
    #Train the model over 10 epochs
    print("created LSTM sentence model, training")
    for epoch in xrange(10):
        start=time.time()
        print("->epoch: ", epoch)
        for i in xrange(numSentences):
            # Convert sentence to dense
            newSentence = []
            for token in sentences[i]:
                if token in word_model:
                    newSentence.append(word_model[token])
            # Only process sentences with more than one work/token
            if len(newSentence) > 1:
                # Input is sentence, padded with zero vectors to fill last word window
                # Shape (numWords (rounded up to nearest time window) + 1, 100)
                x = np.zeros((math.ceil(len(newSentence)/timeWindow)*timeWindow+1,100),dtype=np.float32)
                x[0:len(newSentence)][:] = np.asarray(newSentence)
                x_in = np.zeros((1,timeSteps,100),dtype=np.float32)
                x_out = np.zeros((1,timeSteps,100),dtype=np.float32)
                # Previous hidden state at start of each sentence set to zeros
                hidden = np.zeros((100),dtype=np.float32)
                depth = 0
                # Process sentence in blocks specified by time window
                # End of sentence and is not used as input
                while depth < (len(newSentence)-1):
                    # First part of input for each window is previous hidden state
                    x_in[0][0][:]=hidden
                    # Remainder of input is the next words in the sentence
                    x_in[0][1:timeSteps][:]=x[depth:(depth+timeWindow)][:]
                    # Expected output is next token in input
                    x_out[0][:][:] = x[depth:(depth+timeWindow+1)][:]
                    # Train on batch size 1,
                    model.train_on_batch(x_in,x_out)
                    # Get the hidden state after the last word
                    hidden = get_activations(x_in)[0][timeWindow][:]
                    # Update depth in sentence
                    depth += timeWindow
            if (i%100 == 0):
                print ("epoch: ",epoch,", ",i / numSentences * 100)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
    #Store the model architecture to a file
    json_string = model.to_json()
    open('./model_files/reports.rnn_sentence_architecture.json', 'w').write(json_string)
    model.save_weights('./model_files/reports.rnn_sentence_weights.h5',overwrite=True)
    print("Trained sentence model")

# Predicts the next words after the given input phrase
# Input is a string and the number of words to predict (default is 5)
# No return, however the word vectors and predictions are printed
def nextWords(phrase,numWords=5):
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence model")
    model = model_from_json(open('./model_files/reports.rnn_sentence_architecture.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    print("RNN sentence model loaded")

    phrase = textPreprocess(phrase)[0]

    newPhrase=[]
    for token in phrase:
        if token in word_model:
            newPhrase.append(word_model[token])
    while (numWords>0):
        x=np.asarray([newPhrase])
        nextWord = model.predict(x,batch_size=1)
        newPhrase.append(nextWord[0][len(newPhrase)-1])
        numWords -= 1
    print("words in vector format are:")
    print(newPhrase)
    print("sentence is:")
    for word in newPhrase:
        print(word_model.most_similar(positive=[word],topn=1)[0][0])

# Converts the sentence predictor model to a sentence encoder only model
# Required to be able to generate sentence vectors
def sentenceToEncoder():
    maxLen = 50
    print("loading RNN sentence model")
    full = model_from_json(open('./model_files/reports.rnn_sentence_architecture.json').read())
    full.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    print("RNN sentence model loaded")

    print('building sentence Endocer model...')
    m = Sequential()
    m.add(LSTM(100, input_length=maxLen-1, input_dim=100, weights=full.layers[0].get_weights()))
    m.compile(loss='mse', optimizer='adam')
    print("created sentence Encoder model")

    #Store the encoder model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_sentence_encoder.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_sentence_encoder_weights.h5',overwrite=True)

    print("Encoder sentence model saved")

# Generate the dense representations of each sentence and store them to a file
# Required to be able to find similar sentences
def sentencesToDense():
    # Maximum sentence length
    maxLen = 50
    print("loading sentences")
    sentences = getProcessedSentences()
    numSentences = len(sentences)
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence encoder")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("loaded RNN sentence model encoder")
    predictions = np.zeros((numSentences,100))
    for i in xrange(numSentences):
        # Convert sentence to dense
        newSentence = []
        for token in sentences[i]:
            if token in word_model:
                newSentence.append(word_model[token])
        # Trim sentences greater than maxLen
        newSentence = newSentence[:maxLen-1]
        # Delete last word from sentence to make input of the same format as training
        newSentence = newSentence[:len(newSentence)-1]
        # Store the sentence vector
        if len(newSentence) > 0:
            # Convert the sentence to an array, note that the length is variable (unlike training)
            x=np.asarray([newSentence])
            predictions[i] = model.predict(x,batch_size=1)
        # Print processing percentage
        if ((i%1000)==0):
            print (i / numSentences * 100)
    file = open('./model_files/reports_rnn_sentences', 'w')
    pickle.dump(predictions, file)
    file.close()

# Converts the search term to a sentence vector
def getSearchTerm(searchTerm):
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence encoder")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("loaded RNN sentence model encoder")
    newTerm=[]
    for token in searchTerm:
        if token in word_model:
            newTerm.append(word_model[token])
    # Use the whole term as the end of sentence token is unlikely to be included
    x=np.asarray([newTerm])
    searchTerm = model.predict(x,batch_size=1)
    return searchTerm

# Returns the most similar sentences to the given search term
# Search term is a sentence vector
def most_similar(searchTerm,topn=5):
    print("loading sentence vectors")
    file = open('./model_files/reports_rnn_sentences', 'r')
    predictions = pickle.load(file)
    file.close()
    print("loaded sentence vectors")
    distance = cdist(searchTerm,predictions,'cosine')
    idx = np.argsort(distance)
    results = []
    i = 0
    while (i < topn):
        index = idx[0][i]
        result = []
        result.append(index)
        result.append(1-distance[0][index])
        results.append(result)
        i = i + 1
    return results

# Returns the least similar sentences to the given search term
# Search term is a sentence vector
def least_similar(searchTerm,topn=5):
    print("loading sentence vectors")
    file = open('./model_files/reports_rnn_sentences', 'r')
    predictions = pickle.load(file)
    file.close()
    print("loaded sentence vectors")
    similarity = 1-cdist(searchTerm,predictions,'cosine')
    idx = np.argsort(similarity)
    results = []
    i = 0
    while (i < topn):
        index = idx[0][i]
        result = []
        result.append(index)
        result.append(similarity[0][index])
        results.append(result)
        i = i + 1
    return results

# Returns the similar sentences to the given search term below a given threshold
# Search term is a sentence vector and a similarity upperbound
def similar(searchTerm,upperbound=0.99,topn=5):
    print("loading sentence vectors")
    file = open('./model_files/reports_rnn_sentences', 'r')
    predictions = pickle.load(file)
    file.close()
    print("loaded sentence vectors")
    distance = cdist(searchTerm,predictions,'cosine')
    idx = np.argsort(distance)
    results = []
    i = 0
    offset = 0
    while ((1-distance[0][idx[0][offset]]) > upperbound):
        offset += 1
    while (i < topn):
        index = idx[0][i+offset]
        result = []
        result.append(index)
        result.append(1-distance[0][index])
        results.append(result)
        i = i + 1
    return results

# Returns the cosine distance between the two given sentences
# Inputs are strings
def compareSentences(sentence1,sentence2):
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence encoder")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("loaded RNN sentence model encoder")
    print("First sentence is: ",sentence1)
    sentence1=textPreprocess(sentence1)[0]
    print(sentence1)
    print("Second sentence is: ",sentence2)
    sentence2=textPreprocess(sentence2)[0]
    print(sentence2)
    newSentence=[]
    for token in sentence1:
        if token in word_model:
            newSentence.append(word_model[token])
    # Use the whole term as the end of sentence token is unlikely to be included
    x=np.asarray([newSentence])
    sentence1 = model.predict(x,batch_size=1)
    newSentence=[]
    for token in sentence2:
        if token in word_model:
            newSentence.append(word_model[token])
    # Use the whole term as the end of sentence token is unlikely to be included
    x=np.asarray([newSentence])
    sentence2 = model.predict(x,batch_size=1)
    similarity = 1- cdist(sentence1,sentence2,'cosine')
    print("Similarity is: ",similarity)
    return similarity

# Searches for and returns the 5 most similar and least similar sentences to the search term
# Search term is a string
def searchRNN(searchTerm):
    # preprocess the search term to make it of the same format as the sentences
    searchTerm = textPreprocess(searchTerm)[0]
    print("Searching for: ")
    print(searchTerm)
    # Convert the search term to a sentence vector
    searchTerm_rnn = getSearchTerm(searchTerm)
    print(searchTerm_rnn)
    # Find the most similar sentences
    similarSentences = most_similar(searchTerm_rnn,topn=20)
    # Find the least similar sentences
    dissimilarSentences = least_similar(searchTerm_rnn,topn=20)
    # Inform the user if the search term was invalid
    if (similarSentences == []):
    	print ("ERROR: Invalid search term")

    print("loading sentences")
    sentences = getProcessedSentences()
    print("loaded sentences")

    for sentenceIdx in similarSentences:
    	print("----------Most similar")
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

    for sentenceIdx in dissimilarSentences:
    	print("----------Least similar")
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

# Converts the preprocessed reports to sets of sentence vectors
# Saves the new reports to a file
def reportsToDense():
    # Max number of words in sentence, detemined in the initial report processing
    maxLen = 50
    reportLen = 0
    reports = getProcessedReports()
    for report in reports:
        length = len(report)
        if length > reportLen:
            reportLen = length
    print("longest report has ",reportLen," sentences.")
    print("loading RNN sentence encoder model")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("RNN sentence encoder model loaded")

    denseReports=[]

    for i in xrange(len(reports)):
        denseReport = []
        for sentence in reports[i]:
            newSentence = []
            for token in sentence:
                if token in word_model:
                    newSentence.append(word_model[token])
            # Trim sentences greater than maxLen
            newSentence = newSentence[:maxLen-1]
            # Delete last word from sentence to make input of the same format as training
            newSentence = newSentence[:len(newSentence)-1]
            # Store the sentence vector
            if len(newSentence) > 0:
                # Convert the sentence to an array, note that the length is variable (unlike training)
                x=np.asarray([newSentence])
                denseReport.append(model.predict(x,batch_size=1))
        denseReports.append(denseReport)
        if ((i%100)==0):
            print (i / numSentences * 100)
    file = open('./model_files/reports_dense', 'w')
    pickle.dump(denseReports, file)
    file.close()
    print("dense reports saved")

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
