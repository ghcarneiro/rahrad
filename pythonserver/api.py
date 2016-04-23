from __future__ import division
import gensim, logging
import sys
import math
from pprint import pprint   # pretty-printer
import pickle
import re
from nltk.corpus import stopwords
from nltk import stem
import nltk
import time
import datetime
import os
import numpy as np

from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

# Load model files
tfidf_model = gensim.models.TfidfModel.load('../website/model_files/reports.tfidf_model')
lsi_model = gensim.models.LsiModel.load('../website/model_files/reports.lsi_model')
dictionary = gensim.corpora.Dictionary.load('../website/model_files/reports.dict')
print 'Models loaded'

# set of stop words
stop = set()
# specialist dictionary
medical = dict()

#load set of stop words
negations = set(('no', 'nor','against','don', 'not'))
stop = set(stopwords.words("english")) - negations

#load dictionary of specialist lexicon
file = open('../website/dictionary_files/medical.pkl', 'r')
medical = pickle.load(file)
file.close()

# runs the preprocessing procedure to the supplied text
# input is string of text to be processed
# output is the same string processed
def textPreprocess(text,minimal=False):


	if not minimal:
		text = re.sub("[^a-zA-Z\-]"," ",text) # remove non-letters, except for hyphens
		text = text.lower() # convert to lower-case
		text = text.split() # tokenise string
		text = [word for word in text if len(word) > 1] # remove all single-letter words
		# remove stop words
		text = [word for word in text if not word in stop]
	else:
		# Alterative Minimal processing, lowercase and keep punctuation
		text = text.lower()
		text = re.split("([^\w\-]+)||\b", text)
		text = [word.replace(' ','') for word in text]
		text = filter(None, text)

	#look up variable length sequences of words in medical dictionary, stem them if not present
	numTokens = 5 #phrases up to 5 words long
	while numTokens > 0:
		processedText=[]
		start=0
		#Check each phrase of n tokens while there are sufficient tokens after
		while start < (len(text) - numTokens):
			phrase=text[start]
			nextToken=1
			while nextToken < numTokens:
				#add the next tokens to the current one
				phrase = phrase+" "+text[start+nextToken]
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
				processedText.append(text[start])
				start += 1
		#Keep remaining tokens without enough tokens after them
		while start < len(text):
			processedText.append(text[start])
			start += 1
		text = processedText
		numTokens -= 1

	# word stemming (list of word stemmers: http://www.nltk.org/api/nltk.stem.html)
	# text = [stem.snowball.EnglishStemmer().stem(word) for word in text]
	# text = [stem.PorterStemmer().stem(word) for word in text]

	return(text)

# get all the derivations of each word in the search term, and generates a new search term based on these derivations (only if they exist in the dictionary)
# input is the search term to use
# output is a new search term that contains all of the derivations
def getDerivations(searchTerm):
	newSearchTerm = []
	for word in searchTerm:
		for i in range(len(word)):
			if i == 0:
				if word in dictionary.values():
					newSearchTerm.append(word)
			else:
				if word[:-i] in dictionary.values():
					newSearchTerm.append(word[:-i])
	return newSearchTerm

def runReportSimilarity(fileName,fileName2,threshold=0.9,reportType="lsi"):
    """ Assumes reports have FINDINGS: or REPORT: """
    return_sentences = ','  
    wordsToFind = ["FINDINGS:","REPORT:"]
    report1 = fileName
    report2 = fileName2

    startLoc1 = -1
    startLoc2 = -1
    for word in wordsToFind:
        if startLoc1 == -1 and report1.find(word) != -1:
            startLoc1 = report1.find(word)+len(word)
        else:
            startLoc1 = 0
        # In case trainee doesn't write findings/report
        if startLoc2 == -1 and report2.find(word) != -1:
            startLoc2 = report2.find(word)+len(word)
        else:
            startLoc2 = 0

    sCom = []

    report1 = report1[startLoc1:]
    report2 = report2[startLoc2:]
    sentences1 = report1.split('.')
    sentences2 = report2.split('.')
    sent1 = sentences1[:]
    sent2 = sentences2[:]

    if reportType == "lsi":
        report1 = textPreprocess(report1)
        report1 = getDerivations(report1)
        report2 = textPreprocess(report2)
        report2 = getDerivations(report2)
        for i in range(len(sentences1)):
            sentences1[i] = textPreprocess(sentences1[i])
            sentences1[i] = getDerivations(sentences1[i])
        for i in range(len(sentences2)):
            sentences2[i] = textPreprocess(sentences2[i])
            sentences2[i] = getDerivations(sentences2[i])

		#corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
        vec_lsi1 = lsi_model[tfidf_model[dictionary.doc2bow(report1)]]
        vec_lsi2 = lsi_model[tfidf_model[dictionary.doc2bow(report2)]]
        sen1Corp = [dictionary.doc2bow(sent) for sent in sentences1]
        sen2Corp = [dictionary.doc2bow(sent) for sent in sentences2]
        vec_lsis1 = lsi_model[tfidf_model[sen1Corp]]
        vec_lsis2 = lsi_model[tfidf_model[sen2Corp]]

		# print corpus.num_terms
		# ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=corpus.num_terms)
        ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=10)
		# similarity table
        for  i in vec_lsis2:
            sCom.append(ind[i])
    elif reportType == "rnn":
        sCom = rnn.compareReportSentences(report1,report2)
        sCom2 = []
        for i in range(len(sCom[0])):
            row = []
            for j in range(len(sCom)):
                row.append(sCom[j][i])
            sCom2.append(row)
				
        sCom = sCom2

    missing = [0 for s in sent1]
	# obtain correct sentence
    i = 0

    for col in sCom:
	#for col in range(len(sCom[0]))
	#for col in sent2: 
        aboveTopThreshold = False 
        j = 0
        bestSim = 0
        for sim in col:
            if sim > threshold: 
                aboveTopThreshold = True 
            if sim > bestSim:
                bestSim = sim
            if missing[j] < sim:
                missing[j] = sim
				
            j+=1

        threshold_value = bestSim

        if aboveTopThreshold and (sent1[i].strip()) != "":
            s="n\t"+sent2[i]+" "+str(threshold_value)+"\t"+str(i) + "\n"	
            print s
            return_sentences = return_sentences + str(i) + ','
        elif (threshold_value > 0) and (sent1[i].strip()) != "": # Exclude blank strings
			#sent2[i] = " ".join([k for k in sent2[i]])
            s ="e\t"+sent2[i]+" "+str(threshold_value)+"\t"+str(i) + "\n"
            print s
        i+=1
    i=0
    return_sentences = return_sentences + '-100,'
    for k in missing:
        if k <= threshold and (sent1[i].strip()) != "":
			#sent1[i] = " ".join([k for k in sent1[i]])
			#s = str(k)
            s = "m\t"+sent1[i]+"\t"+str(i) + "\n"
            print s
            return_sentences = return_sentences + str(i) + ','
        elif (sent1[i].strip()) != "": # Checks that string is not empty
	    s = "t\t"+sent1[i]+"\t"+str(i) + "\n"
            print s	
        i+=1
			
    return return_sentences

##############################################
#      HTTP request handler
##############################################

class ReportSimilarity(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('expert_report', type=str, help='Expert report id')
            parser.add_argument('learner_report', type=str, help='Learner report')
            args = parser.parse_args()
            expertreport = args['expert_report']
            learnerreport = args['learner_report']
            learnerresult = runReportSimilarity(expertreport, learnerreport)
            return learnerresult
        except Exception as e:
            return {'error': str(e)}

api.add_resource(ReportSimilarity, '/')

if __name__ == '__main__':
    app.run(debug=True)
