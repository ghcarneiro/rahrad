import gensim
import sys
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk import stem
import nltk
import re

# global variables, loaded during first call to text preprocessing
# set of stop words
stop = set()
medical = dict()

dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
print "loaded reports"

fullreport1 = []
fullreport2 = []

def textPreprocessDerivations(text,reportType,minimal=True):
	global medical
	if not medical:
		#load dictionary of specialist lexicon
		file = open('./dictionary_files/medical.pkl', 'r')
		medical = pickle.load(file)
		print "loaded medical"
		file.close()
	#load set of stop words
	global stop
	if not stop:
		negations = set(('no', 'nor','against','don', 'not'))
		stop = set(stopwords.words("english")) - negations

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
                print text
                print "minimal processing done"

	#look up variable length sequences of words in medical dictionary, stem them if not present
	numTokens = 3 #phrases up to 5 words long
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

	newSearchTerm = []
	for word in text:
		for i in range(len(word)):
			if i == 0:
				if word in dictionary.values():
					newSearchTerm.append(word)
                                        reportType.append(word)
			else:
				if word[:-i] in dictionary.values():
					newSearchTerm.append(word[:-i])
                                        reportType.append(word[:-i])
	return newSearchTerm

def runReportSimilarity(fileName,threshold=0.9,reportType="lsi"):
    """ Assumes reports have FINDINGS: or REPORT: """
    fileText = [row.rstrip('\n') for row in open(fileName)]
                
    wordsToFind = ["FINDINGS:","REPORT:"]
    report1 = fileText[0]
    report2 = fileText[1]

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
        for i in range(len(sentences1)):
            sentences1[i] = textPreprocessDerivations(sentences1[i],fullreport1)
            print "sentence 1 derivations done"
        for i in range(len(sentences2)):
            sentences2[i] = textPreprocessDerivations(sentences2[i],fullreport2)
            print "sentence 2 derivations done"

        print fullreport1
        print fullreport2

		#corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
        tfidf_model = gensim.models.TfidfModel.load('./model_files/reports.tfidf_model')
        print "tf-idf loaded"
        lsi_model = gensim.models.LsiModel.load('./model_files/reports.lsi_model')
        print "lsi loaded"
        vec_lsi1 = lsi_model[tfidf_model[dictionary.doc2bow(fullreport1)]]
        vec_lsi2 = lsi_model[tfidf_model[dictionary.doc2bow(fullreport2)]]
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
        pythonpleaser = "3"

    missing = [0 for s in sent1]
	# obtain correct sentence
    i = 0


    output = {'missing': 0, 'corrections': 0, 'extras': 0, 'correct': 0}
    for col in sCom:
	#for col in range(len(sCom[0]))
	#for col in sent2: 
        threshold_value = 0
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
   
        if aboveTopThreshold: 
			#maybe add percentage for debugging
			#sent2[i] = " ".join([k for k in sent2[i]])
            s="n\t"+sent2[i]+" "+str(threshold_value)+"\t"+str(i)	
            output['correct'] += 1
            print s
        elif (threshold_value > 0): # Exclude blank strings
			#sent2[i] = " ".join([k for k in sent2[i]])
            s ="e\t"+sent2[i]+" "+str(threshold_value)+"\t"+str(i)
            output['extras'] += 1
            print s
        i+=1
    i=0
    for k in missing:
        if k <= threshold:
			#sent1[i] = " ".join([k for k in sent1[i]])
			#s = str(k)
            s = "m\t"+sent1[i]+"\t"+str(i)
            output['missing'] += 1
            print s
        else:
	    s = "t\t"+sent1[i]+"\t"+str(i)
            print s		
        i+=1
			
    return output
			
if __name__ == '__main__':
	# ./similarity input.txt threshold
	# or default threshold
	if (len(sys.argv) < 2):
                print("ERROR: Please specify an input file")
                sys.exit()

	fileName = str(sys.argv[1])
	runReportSimilarity(fileName)
