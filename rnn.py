from __future__ import division
from __future__ import print_function
import pickle
import csv
import re
from nltk import stem
import nltk
import gensim
import numpy as np
# import klepto
from scipy.spatial.distance import cdist
import preprocess
import keras
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from sklearn import cross_validation
import theano
import math
import decimal
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

# Global variables for use in RNNs
# Number of sentence hidden units / sentence vector size
SENTENCE_HIDDEN = 300
# Number of report hidden units / report vector size
REPORT_HIDDEN = 1000
# Number of sentences to process in each batch
BATCH_SIZE = 128
# specialist dictionary, loaded during first call to text preprocessing
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
    # text.append(["end_rep"])

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
            reports[i] = textPreprocess(reports[i])
            allSentences = allSentences + reports[i]
            if (i%100==0):
                print (i / len(reports) * 100)
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
# output is a list containing the processed reports
def getProcessedReports():
	file = open('./model_files/reports_full', 'r')
	reports = pickle.load(file)
	file.close()

	return reports

# retrieves all sentences that have been preprocessed
# output is a list containing the processed sentences
def getProcessedSentences():
	file = open('./model_files/reports_sentences_full', 'r')
	sentences = pickle.load(file)
	file.close()

	return sentences

# Initialised the epoch training file with the number of epochs
# Input is number of epochs, epoch completed, epoch completion time
def writeStatusHeader(epochs,continueTraining,mode="sentence"):
    if continueTraining:
        if mode is "sentence":
            file = open('./model_files/rnn_sentence_status.txt', 'a')
        else:
            file = open('./model_files/rnn_report_status.txt', 'a')
    else:
        if mode is "sentence":
            file = open('./model_files/rnn_sentence_status.txt', 'w')
        else:
            file = open('./model_files/rnn_report_status.txt', 'w')
    file.write("Running training on "+str(epochs)+" epochs"+"\n")
    file.close()

# Writes the epoch training status to a file
# Input is number of epochs, epoch completed, epoch completion time
def writeStatusEpoch(epochs,currentEpoch,epochTime,mode="sentence"):
    if mode is "sentence":
        file = open('./model_files/rnn_sentence_status.txt', 'a')
    else:
        file = open('./model_files/rnn_report_status.txt', 'a')
    file.write("\t"+"Epoch training time of:"+str(epochTime)+"\n")
    file.write("Completed epoch "+str(currentEpoch)+"/"+str(epochs)+"\n")
    file.close()

# Writes the bucket training status to a file
# Input is current epoch, bucket size and bucket training error rate
def writeStatusBucket(currentEpoch,bucketSize,errorRate,mode="sentence"):
    if mode is "sentence":
        file = open('./model_files/rnn_sentence_status.txt', 'a')
    else:
        file = open('./model_files/rnn_report_status.txt', 'a')
    file.write("\t"+"Epoch "+str(currentEpoch)+" bucket <"+str(bucketSize)+": "+str(errorRate)+"\n")
    file.close()

# Writes the training status to a file
# Input is number of epochs, epoch completed, epoch time and last training error rate
def writeEpochStatus(epochs,currentEpoch,epochTime):
    if currentEpoch == 0:
        file = open('./model_files/rnn_status.txt', 'w')
    else:
        file = open('./model_files/rnn_status.txt', 'a')
    file.write("Completed epoch "+str(currentEpoch)+"/"+str(epochs)+"\n")
    file.write("->Epoch training time of:"+str(epochTime)+"\n")
    file.close()

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

# Loads the preprocessed sentences and buckets them into increments of the given size
# Input is the size of the bucket increments
# Returns the bucketed processed sentences
def getBucketedSentences(bucketSize=10):
    sentences = getProcessedSentences()
    buckets=[]
    # Find the maximum sentence length
    maxLen = 0
    for sentence in sentences:
        if len(sentence) > maxLen:
            maxLen = len(sentence)
    # Create buckets of the specified size up to the maximum sentence length
    while maxLen > 0:
        buckets.append([])
        maxLen -= bucketSize
    # Place the sentences into their buckets
    for sentence in sentences:
        index = int(len(sentence)/bucketSize)
        buckets[index].append(sentence)
    # Remove empty buckets
    buckets = filter(None, buckets)
    return buckets

# Build an RNN on sentences using buckets in increments of the speficied size
# By default train a new model of 10 epochs with bucket increments of 10
# Input is number of training epochs, whether to continue training an existing model and the size of bucket increments
def buildSentenceRNN(epochs=10,continueTraining=False,bucketSize=10):
    print("loading and bucketing sentences")
    buckets = getBucketedSentences(bucketSize)
    print("sentences loaded and bucketed")
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    # Continue training the existing model if requested
    if continueTraining:
        print('loading LSTM model...')
        m = model_from_json(open('./model_files/reports.rnn_sentence_architecture.json').read())
        m.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    # Otherwise create model before training it
    else:
        print('building LSTM model...')
        m = Sequential()
        m.add(LSTM(SENTENCE_HIDDEN, input_dim=100, return_sequences=True))
        m.add(LSTM(100, return_sequences=True))
        m.compile(loss='mse', optimizer='adam')
        #Store the model architecture to a file
        json_string = m.to_json()
        open('./model_files/reports.rnn_sentence_architecture.json', 'w').write(json_string)
    #Train the model over 10 epochs
    print("created LSTM sentence model")
    print("training...")
    writeStatusHeader(epochs,continueTraining)
    for epoch in xrange(epochs):
        start=time.time()
        print("Starting Epoch ", epoch," of ",epochs)
        for bucket in buckets:
            # Find current bucket size
            currentSize = (int(len(bucket[0])/bucketSize)+1)*bucketSize
            numSentences = len(bucket)
            bucketError = 0
            bucketBatches = math.ceil(numSentences/BATCH_SIZE)
            for i,sentence in enumerate(bucket):
                # Create batch in memory
                if ((i% BATCH_SIZE) == 0):
                    batch = np.zeros((BATCH_SIZE,currentSize-1,100),dtype=np.float32)
                    expected = np.zeros((BATCH_SIZE,currentSize-1,100),dtype=np.float32)
                # Convert sentence to dense
                newSentence = []
                for token in sentence:
                    if token in word_model:
                        newSentence.append(word_model[token])
                # Store sentence in batch
                if len(newSentence) > 0:
                    temp = np.asarray(newSentence)
                    # Last word of sentence is not used as input
                    batch[i%BATCH_SIZE][0:len(newSentence)-1][:]=temp[0:len(newSentence)-1][:]
                    # First word of sentence is not used as expected output (right shifted input)
                    expected[i%BATCH_SIZE][0:len(newSentence)-1][:]=temp[1:len(newSentence)][:]
                # Train on batch
                if (((i+1)% BATCH_SIZE) == 0):
                    error = m.train_on_batch(batch,expected)
                    bucketError += error[0]
                    if ((i+1) % (BATCH_SIZE*10)==0):
                        print("epoch: ",epoch,", bucket: ",currentSize,", ",i / numSentences * 100)
                        print("-> error of ",error[0])
                # Train on remaining data that does not fit in batch
                elif (i == (numSentences-1)):
                    batch = batch[0:(i+1)%BATCH_SIZE][:][:]
                    expected = expected[0:(i+1)%BATCH_SIZE][:][:]
                    error = m.train_on_batch(batch,expected)
                    bucketError += error[0]
                    print("epoch: ",epoch,", bucket: ",currentSize,", ",i / numSentences * 100)
                    print("-> error of ",error[0])
            bucketError = bucketError/bucketBatches
            print("Epoch ",epoch," with bucket size ",currentSize," had an average error of ",bucketError)
            writeStatusBucket(epoch+1,currentSize,bucketError)
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
        print("updating weights file")
        m.save_weights('./model_files/reports.rnn_sentence_weights.h5',overwrite=True)
        writeStatusEpoch(epochs,epoch+1,end-start)
    print("Trained sentence model")

# Build a RNN on sentences using a fixed window size and no batches
# Due to the updating of weights in keras this function is unusable due to computation time required
def buildSentenceRNNWindowed(epochs=10,continueTraining=False):
    # Max number of words in training set, detemined in the initial report processing
    timeWindow = 10
    timeSteps = timeWindow + 1
    trimmed = 0
    print("loading sentences")
    sentences = getProcessedSentences()
    numSentences = len(sentences)
    print("loaded sentences")
    print("splitting data")
    # Shuffle and split the data set - only use 5% in each epoch
    epoch_split = cross_validation.ShuffleSplit(numSentences, n_iter=epochs,test_size=.95)
    training_size = 0.05 * numSentences
    print("split data")
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print('building LSTM model...')
    model = Sequential()
    model.add(LSTM(SENTENCE_HIDDEN, input_length=timeSteps, input_dim=100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.compile(loss='mse', optimizer='adam')
    if continueTraining:
        model.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    #Store the model architecture to a file
    json_string = model.to_json()
    open('./model_files/reports.rnn_sentence_architecture.json', 'w').write(json_string)
    get_activations = theano.function([model.layers[0].input], model.layers[0].get_output(train=False), allow_input_downcast=True)
    #Train the model over 10 epochs
    print("built LSTM sentence model, training")
    for epoch,[training_set,test_set] in enumerate(epoch_split):
        start=time.time()
        print("Starting epoch: ", epoch," of ",epochs)
        for progress,index in enumerate(training_set):
            # Convert sentence to dense
            newSentence = []
            for token in sentences[index]:
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
                    history = model.train_on_batch(x_in,x_out)
                    # Get the hidden state after the last word
                    hidden = get_activations(x_in)[0][timeWindow][:]
                    # Update depth in sentence
                    depth += timeWindow
            if (progress%100 == 0):
                print("epoch ",epoch,"/",epochs,": ",progress / training_size * 100,"%")
                print("->Training error rate is: ",history[0])
        end = time.time()
        print("Epoch ",epoch," took ",end-start," seconds")
        model.save_weights('./model_files/reports.rnn_sentence_weights.h5',overwrite=True)
        writeStatus(epochs,epoch,end-start,history[0])
        print("Updated weights and status files")
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
    print("loading RNN sentence model")
    full = model_from_json(open('./model_files/reports.rnn_sentence_architecture.json').read())
    full.load_weights('./model_files/reports.rnn_sentence_weights.h5')
    print("RNN sentence model loaded")

    print('building sentence Encoder model...')
    m = Sequential()
    m.add(LSTM(SENTENCE_HIDDEN, input_dim=100, weights=full.layers[0].get_weights()))
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
    predictions = np.zeros((numSentences,SENTENCE_HIDDEN))
    for i in xrange(numSentences):
        # Convert sentence to dense
        newSentence = []
        for token in sentences[i]:
            if token in word_model:
                newSentence.append(word_model[token])
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
    print("saving dense sentence representations")
    file = open('./model_files/reports_rnn_sentences', 'w')
    np.save(file,predictions)
    file.close()

# Loads the dense sentence vectors from a file
# Returns an matrix of sentence vectors
def loadDenseSentences():
    file = open('./model_files/reports_rnn_sentences', 'r')
    sentences = np.load(file)
    file.close()
    return sentences

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
    predictions = loadDenseSentences()
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
    predictions = loadDenseSentences()
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
    predictions = loadDenseSentences()
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

# Returns the cosine distance between the two given sentences
# Inputs are strings
def compareReportSentences(report1,report2):
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence encoder")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("loaded RNN sentence model encoder")
    report1=textPreprocess(report1)
    report2=textPreprocess(report2)
    # Convert report one to dense sentences
    tempReport=[]
    for i,sentence in enumerate(report1):
        newSentence=[]
        for token in sentence:
            if token in word_model:
                newSentence.append(word_model[token])
        # Store the sentence vector
        if len(newSentence) > 0:
            # Convert the sentence to an array, note that the length is variable (unlike training)
            x=np.asarray([newSentence])
            tempReport.append(model.predict(x,batch_size=1)[0][:])
        else:
            tempReport.append(np.zeros(SENTENCE_HIDDEN))
    report1=np.asarray(tempReport)
    # Convert report two to dense sentences
    tempReport=[]
    for j,sentence in enumerate(report2):
        newSentence=[]
        for token in sentence:
            if token in word_model:
                newSentence.append(word_model[token])
        # Store the sentence vector
        if len(newSentence) > 0:
            # Convert the sentence to an array, note that the length is variable (unlike training)
            x=np.asarray([newSentence])
            tempReport.append(model.predict(x,batch_size=1)[0][:])
        else:
            tempReport.append(np.zeros(SENTENCE_HIDDEN))
    report2=np.asarray(tempReport)
    similarity = 1- cdist(report1,report2,'cosine')
    return similarity

# Searches for and returns the 20 most similar and least similar sentences to the search term
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
    # Find the most similar sentences less than 99.99% similar
    similarSentences9999 = similar(searchTerm_rnn,upperbound=0.9999,topn=20)
    # Find the most similar sentences less than 99% similar
    similarSentences9900 = similar(searchTerm_rnn,upperbound=0.99,topn=20)
    # Find the least similar sentences
    dissimilarSentences = least_similar(searchTerm_rnn,topn=20)
    # Inform the user if the search term was invalid
    if (similarSentences == []):
    	print ("ERROR: Invalid search term")

    print("loading sentences")
    sentences = getProcessedSentences()
    print("loaded sentences")

    print("----------Most similar")
    for sentenceIdx in similarSentences:
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

    print("----------99.99 similar")
    for sentenceIdx in similarSentences9999:
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

    print("----------99.00 similar")
    for sentenceIdx in similarSentences9900:
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

    print("----------Least similar")
    for sentenceIdx in dissimilarSentences:
    	print("Sentence #: " + str(sentenceIdx[0]) + " Similarity: " + str(sentenceIdx[1]) )
    	print(sentences[sentenceIdx[0]])

# Converts the preprocessed reports to sets of sentence vectors
# Saves the new reports to a file
def reportsToDense():
    print("loading reports")
    reports = getProcessedReports()
    numReports=len(reports)
    print("reports loaded")
    print("loading word2vec model")
    word_model = gensim.models.Word2Vec.load("./model_files/reports.word2vec_model")
    print("loaded word2vec model")
    print("loading RNN sentence encoder model")
    model = model_from_json(open('./model_files/reports.rnn_sentence_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_sentence_encoder_weights.h5')
    print("RNN sentence encoder model loaded")

    endToken = np.ones((SENTENCE_HIDDEN),dtype=np.float32)

    denseReports=[]
    print("Converting reports")
    for i in xrange(len(reports)):
        denseReport = []
        for sentence in reports[i]:
            newSentence = []
            for token in sentence:
                if token in word_model:
                    newSentence.append(word_model[token])
            # Delete last word from sentence to make input of the same format as training
            newSentence = newSentence[:len(newSentence)-1]
            # Store the sentence vector
            if len(newSentence) > 0:
                # Convert the sentence to an array, note that the length is variable (unlike training)
                x=np.asarray([newSentence])
                denseReport.append(model.predict(x,batch_size=1)[0][:])
        denseReport.append(endToken)
        denseReports.append(denseReport)
        if ((i%100)==0):
            print (i / numReports * 100)
    file = open('./model_files/reports_rnn_dense', 'w')
    pickle.dump(denseReports, file)
    file.close()
    print("dense reports saved")

# retrieves all reports in their dense vector format
# output is a list containing the reports a lists of dense sentence vectors
def getDenseReports():
    file = open('./model_files/reports_rnn_dense', 'r')
    denseReports = pickle.load(file)
    file.close()
    return denseReports

# Loads the dence reports and buckets them into increments of the given size
# Input is the size of the bucket increments
# Returns the bucketed dense reports
def getBucketedReports(bucketSize=10):
    reports = getDenseReports()
    buckets=[]
    # Find the maximum report length
    maxLen = 0
    for report in reports:
        if len(report) > maxLen:
            maxLen = len(report)
    # Create buckets of the specified size up to the maximum sentence length
    while maxLen > 0:
        buckets.append([])
        maxLen -= bucketSize
    # Place the reports into their buckets, only use reports with at least one sentence
    for report in reports:
        if len(report) > 1:
            index = int(len(report)/bucketSize)
            buckets[index].append(report)
    # Remove empty buckets
    buckets = filter(None, buckets)
    return buckets

# Build an RNN on reports using buckets in increments of the speficied size
# By default train a new model of 10 epochs with bucket increments of 10
# Input is number of training epochs, whether to continue training an existing model and the size of bucket increments
def buildReportRNN(epochs=10,continueTraining=False,bucketSize=10):
    print("loading and bucketing reports")
    buckets = getBucketedReports(bucketSize)
    print("reports loaded and bucketed")
    print('building LSTM reports model...')
    # Continue training the existing model if requested
    if continueTraining:
        m = model_from_json(open('./model_files/reports.rnn_reports_architecture.json').read())
        m.load_weights('./model_files/reports.rnn_reports_weights.h5')
    # Otherwise create model before training it
    else:
        m = Sequential()
        m.add(LSTM(REPORT_HIDDEN, input_dim=SENTENCE_HIDDEN, return_sequences=True))
        m.add(LSTM(SENTENCE_HIDDEN, return_sequences=True))
        m.compile(loss='mse', optimizer='adam')
        #Store the model architecture to a file
        json_string = m.to_json()
        open('./model_files/reports.rnn_reports_architecture.json', 'w').write(json_string)
    #Train the model over 10 epochs
    print("created LSTM reports model")
    print("training...")
    writeStatusHeader(epochs,continueTraining,mode="report")
    for epoch in xrange(epochs):
        start=time.time()
        print("Starting Epoch ", epoch," of ",epochs)
        for bucket in buckets:
            # Find current bucket size
            currentSize = (int(len(bucket[0])/bucketSize)+1)*bucketSize
            numReports = len(bucket)
            bucketError = 0
            bucketBatches = math.ceil(numReports/BATCH_SIZE)
            for i,report in enumerate(bucket):
                # Create batch in memory
                if ((i% BATCH_SIZE) == 0):
                    batch = np.zeros((BATCH_SIZE,currentSize-1,SENTENCE_HIDDEN),dtype=np.float32)
                    expected = np.zeros((BATCH_SIZE,currentSize-1,SENTENCE_HIDDEN),dtype=np.float32)
                # Store report in batch
                temp = np.asarray(report)
                # Last sentence of report is not used as input
                batch[i%BATCH_SIZE][0:len(report)-1][:]=temp[0:len(report)-1][:]
                # First sentence of report is not used as expected output (right shifted input)
                expected[i%BATCH_SIZE][0:len(report)-1][:]=temp[1:len(report)][:]
                # Train on batch
                if (((i+1)% BATCH_SIZE) == 0):
                    error = m.train_on_batch(batch,expected)
                    bucketError += error[0]
                    if ((i+1) % (BATCH_SIZE*10)==0):
                        print("epoch: ",epoch,", bucket: ",currentSize,", ",i / numReports * 100)
                        print("-> error of ",error[0])
                # Train on data that does not fit in batch
                elif (i == (numReports-1)):
                    batch = batch[0:(i+1)%BATCH_SIZE][:][:]
                    expected = expected[0:(i+1)%BATCH_SIZE][:][:]
                    error = m.train_on_batch(batch,expected)
                    bucketError += error[0]
                    print("epoch: ",epoch,", bucket: ",currentSize,", ",i / numReports * 100)
                    print("-> error of ",error[0])
            bucketError = bucketError/bucketBatches
            print("Epoch ",epoch," with bucket size ",currentSize," had an average error of ",bucketError)
            writeStatusBucket(epoch+1,currentSize,bucketError,mode="report")
        end = time.time()
        print("epoch ",epoch," took ",end-start," seconds")
        print("updating weights file")
        m.save_weights('./model_files/reports.rnn_reports_weights.h5',overwrite=True)
        writeStatusEpoch(epochs,epoch+1,end-start,mode="report")
    print("Trained reports model")

# Converts the reports model to a reports encoder only model
# Required to be able to generate report vectors
def reportToEncoder():
    print("loading RNN report model")
    full = model_from_json(open('./model_files/reports.rnn_reports_architecture.json').read())
    full.load_weights('./model_files/reports.rnn_reports_weights.h5')
    print("RNN report model loaded")

    print('building report Encoder model...')
    m = Sequential()
    m.add(LSTM(REPORT_HIDDEN, input_dim=SENTENCE_HIDDEN, weights=full.layers[0].get_weights()))
    m.compile(loss='mse', optimizer='adam')
    print("created report Encoder model")

    #Store the encoder model architecture to a file
    json_string = m.to_json()
    open('./model_files/reports.rnn_report_encoder.json', 'w').write(json_string)
    m.save_weights('./model_files/reports.rnn_report_encoder_weights.h5',overwrite=True)

    print("report encoder model saved")

# Generate the dense vector representations of each report and store them to a file
# Required to be able to find similar reports
def reports2vecs():
    print("loading reports")
    reports = getDenseReports()
    numReports=len(reports)
    print("loaded reports")
    print("loading RNN report encoder")
    model = model_from_json(open('./model_files/reports.rnn_report_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_report_encoder_weights.h5')
    print("loaded RNN report model encoder")
    predictions = np.zeros((numReports,REPORT_HIDDEN))
    for i in xrange(numReports):
        # Convert the report to an array, note that the length is variable (unlike training)
        x=np.asarray([reports[i]])
        predictions[i] = model.predict(x,batch_size=1)
        # Print processing percentage
        if ((i%100)==0):
            print (i / numReports * 100)
    print("saving dense report vector representations")
    file = open('./model_files/reports_rnn_vecs', 'w')
    np.save(file,predictions)
    file.close()

# Loads the dense report vectors from a file
# Returns an matrix of report vectors
def loadReportVecs():
    file = open('./model_files/reports_rnn_vecs', 'r')
    reports = np.load(file)
    file.close()
    return reports

# Converts the search term to a report vector
# Search term is a string
def getReportSearchTerm(searchTerm):
    print("converting report to sentence vector")
    searchTerm = textPreprocess(searchTerm)[0]
    searchTerm = getSearchTerm(searchTerm)
    print("converted report to sentence vector")
    print("loading RNN report encoder")
    model = model_from_json(open('./model_files/reports.rnn_report_encoder.json').read())
    model.load_weights('./model_files/reports.rnn_report_encoder_weights.h5')
    print("loaded RNN report encoder")
    x=np.asarray([searchTerm])
    searchTerm = model.predict(x,batch_size=1)
    return searchTerm

# Returns the most similar reports to the given search term
# Search term is a report vector
def most_similar_reports(searchTerm,topn=5):
    print("loading report vectors")
    predictions = loadReportVecs()
    print("loaded report vectors")
    distance = cdist(searchTerm,predictions,'cosine')
    # Convert the distance to cosine similarity
    # normalise the ocsine similarity to be in the range [0,1]
    similarity = (1-distance[0][:])*.5 + .5
    idx = np.argsort(distance)
    results = []
    i = 0
    while (i < topn):
        index = idx[0][i]
        result = []
        result.append(index)
        result.append(similarity[index])
        results.append(result)
        i = i + 1
    return results
