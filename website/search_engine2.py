# import modules & set up logging
from __future__ import division
import sys
import math
from pprint import pprint   # pretty-printer
import gensim, logging
import pickle
import re
from nltk.corpus import stopwords
from nltk import stem
import nltk
import numpy as np
import time
import datetime
import os
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# global variables, loaded during first call to text preprocessing
# set of stop words
stop = set()
# specialist dictionary
medical = dict()

# runs the preprocessing procedure to the supplied text
# input is string of text to be processed
# output is the same string processed
def textPreprocess(text,minimal=False):
	#load set of stop words
	global stop
	if not stop:
		negations = set(('no', 'nor','against','don', 'not'))
		stop = set(stopwords.words("english")) - negations
	#load dictionary of specialist lexicon
	global medical
	if not medical:
		file = open('./dictionary_files/medical.pkl', 'r')
		medical = pickle.load(file)
		file.close()

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
def getDerivations(searchTerm, dictionary):
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


