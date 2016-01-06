# import modules & set up logging
from __future__ import division
import sys
import math
import random
from pprint import pprint   # pretty-printer
import gensim, logging
import pickle
import csv
import re
from nltk.corpus import stopwords
from nltk import stem
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, svm
from sklearn.metrics import roc_curve
from sklearn.cross_validation import train_test_split

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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


# runs the preprocessing procedure to the supplied text
# input is string of text to be processed
# output is the same string processed
def textPreprocess(text):
	text = re.sub("[^a-zA-Z]"," ",text) # remove non-letters
	text = text.lower() # convert to lower-case
	text = text.split() # tokenise string
	text = [word for word in text if len(word) > 1] # remove all single-letter words

	# remove stop words
	negations = set(('no', 'nor','against','don', 'not'))
	stop = set(stopwords.words("english"))# - negations
	text = [word for word in text if not word in stop]

#TODO - Change stemmer to incorporate radlex
	# word stemming (list of word stemmers: http://www.nltk.org/api/nltk.stem.html)
	text = [stem.snowball.EnglishStemmer().stem(word) for word in text]
	# text = [stem.PorterStemmer().stem(word) for word in text]

	return(text)


# retrieves all fields of all cases, including raw report amongst other fields
# input must be an ARRAY of fileNames. By default, fetches from all case in directory.
# output is a multi-dimensional array containing this data
def getData(fileNames=REPORT_FILES):
	data = []

	for fileName in fileNames:
		with open(fileName,'rb') as file:
			file.readline() # skip header line
			reader = csv.reader(file)
			for row in reader:
				data.append(row)

	return data


# retrieves a list of all reports in its raw unprocessed state
# input must be an ARRAY of fileNames. By default, fetches all reports in directory.
# output is an array containing the reports
def getReports(fileNames=REPORT_FILES):
	reports = []

	for fileName in fileNames:
		with open(fileName,'rb') as file:
			file.readline() # skip header line
			reader = csv.reader(file)
			for row in reader:
				reports.append(row[1])

	return reports

# retrieves all reports that have been preprocessed corresponding to the given diagnoses
# input must be an ARRAY of string specifying the diagnoses to fetched. By default, fetches all reports from all diagnoses.
# output is an array containing the processed reports
def getProcessedReports(diagnoses=DIAGNOSES):
	reports = []

	for diagnosis in diagnoses:
		file = open('./model_files/reports_list_' + diagnosis, 'r')
		reports = reports + pickle.load(file)
		file.close()

	return reports


# determine the total number of case reports in the given files
# input must be an array of fileNames. By default, fetches the total number of case reports in the directory.
# output is the sum of number of case reports
def getNumReports(fileNames=REPORT_FILES):
	data = getData(fileNames)
	return len(data)



# fetches the raw reports, preprocesses them, then saves them as report files
# input must be an array of fileNames to preprocess. By default, preprocesses all reports in directory.
def preprocessReports(fileNames=REPORT_FILES):
	for j in range(len(fileNames)):

		reports = getReports([fileNames[j]])

		print("loading finished")


		for i in xrange(len(reports)):
			print (i / len(reports) * 100)
			reports[i] = textPreprocess(reports[i])

		print("preprocessing finished")

		file = open('./model_files/reports_list_' + DIAGNOSES[j], 'w')
		pickle.dump(reports, file)
		file.close()

		print("report saved")


# builds and saves dictionary and corpus (in BOW form) from report files
def buildDictionary():
	reports = getProcessedReports()

	print("files loaded")

	# build dictionary
	dictionary = gensim.corpora.Dictionary(reports)
	dictionary.save('./model_files/reports.dict')
	print(dictionary)

	print("dictionary finished")

	# build corpus
	corpus = [dictionary.doc2bow(report) for report in reports]
	gensim.corpora.MmCorpus.serialize('./model_files/reports.mm', corpus)
	# print(corpus)

	print("corpus finished")


# NO NEED TO CALL THIS FUNCTION DIRECTLY
# builds and saves the index file used to compute similarity between documents
# input is the corpus file
def build_similarityIndex(corpus):
	index = gensim.similarities.SparseMatrixSimilarity(corpus,num_features=corpus.num_terms)
	index.save('./model_files/reports.index')


# NO NEED TO CALL THIS FUNCTION DIRECTLY
# apply Tf-Idf transformation to generate new model, corpus and index
# input is the corpus file
def transform_tfidf(corpus):
	tfidf_model = gensim.models.TfidfModel(corpus)
	tfidf_model.save('./model_files/reports.tfidf_model')

	newCorpus = tfidf_model[corpus]
	gensim.corpora.MmCorpus.serialize('./model_files/reports_tfidf.mm', newCorpus)
	index = gensim.similarities.SparseMatrixSimilarity(newCorpus,num_features=corpus.num_terms)
	index.save('./model_files/reports_tfidf.index')


# NO NEED TO CALL THIS FUNCTION DIRECTLY
# apply LSI transformation to generate new model, corpus and index
# input is the corpus file and dictionary file
def transform_lsi(corpus,dictionary):
	lsi_model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=10)
	lsi_model.save('./model_files/reports.lsi_model')

	newCorpus = lsi_model[corpus]
	gensim.corpora.MmCorpus.serialize('./model_files/reports_lsi.mm', newCorpus)
	index = gensim.similarities.MatrixSimilarity(newCorpus)
	index.save('./model_files/reports_lsi.index')

# NO NEED TO CALL THIS FUNCTION DIRECTLY
# apply LDA transformation to generate new model, corpus and index
# input is the corpus file and dictionary file
def transform_lda(corpus,dictionary):
	lda_model = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=10)
	lda_model.save('./model_files/reports.lda_model')

	newCorpus = lda_model[corpus]
	gensim.corpora.MmCorpus.serialize('./model_files/reports_lda.mm', newCorpus)
	index = gensim.similarities.MatrixSimilarity(newCorpus)
	index.save('./model_files/reports_lda.index')

# calls the model building and transformation functions to create the model files for the BOW, TFIDF and LSI
def buildModels():
	# load the dictionary
	dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
	print(dictionary)
	# print(dictionary.token2id)

	# load the corpus
	corpus = gensim.corpora.MmCorpus('./model_files/reports.mm')
	# print(corpus)
	print('Example case report under BOW representation: ')
	print(corpus[200])
	# print(list(corpus))

	# build index for similarity comparison using BOW representation
	# build_similarityIndex(corpus)

	# transform model using TFIDF
	# transform_tfidf(corpus)
	tfidf_corpus = gensim.corpora.MmCorpus('./model_files/reports_tfidf.mm')
	print('Example case report under Tf-Idf transformation: ')
	print(list(tfidf_corpus)[200])

	# transform model using LSI
	# transform_lsi(tfidf_corpus,dictionary)
	lsi_corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
	# lsi_model.print_topics()
	print('Example case report under LSI transformation: ')
	print(list(lsi_corpus)[200])

	# transform model using LDA
	transform_lda(corpus,dictionary)
	lda_corpus = gensim.corpora.MmCorpus('./model_files/reports_lda.mm')
	# lda_model.print_topics()
	print('Example case report under LDA transformation: ')
	print(list(lda_corpus)[200])

# function to test the functionality of Word2Vec
def buildWord2VecModel():
	reports = getProcessedReports()

	model = gensim.models.Word2Vec(reports, min_count=3)
	model.init_sims(replace=True)
	model.save("./model_files/reports.word2vec_model")
	print(model)

	# model = gensim.models.Word2Vec.load("zzmodel")
	print("----------------------------------similarity test")
	print(model.similarity("head","brain"))
	print("----------------------------------raw numpy vector of word")
	print(model["age"])
	print("----------------------------------remove outlier")
	print(model.doesnt_match("hours four age".split()))
	print("----------------------------------similar words")
	print(model.most_similar("haem"))

	print("script finished")

# builds and saves the Doc2Vec model of all the processed reports
def buildDoc2VecModel():
	reports = getProcessedReports()

	# construct sentences from reports
	taggedDocuments = []
	for i in range(len(reports)):
		taggedDocument = gensim.models.doc2vec.TaggedDocument(words= reports[i], tags= [i])
		taggedDocuments.append(taggedDocument)


	# model = gensim.models.Doc2Vec(taggedDocuments)
	model = gensim.models.Doc2Vec(size=300, min_count=5, workers=16)

	model.build_vocab(taggedDocuments)

	model.alpha = 0.025 # learning rate

	for epoch in range(10):
		print(epoch)
		model.train(taggedDocuments)
		model.alpha -= 0.001
		model.min_alpha = model.alpha


	model.save("./model_files/reports.doc2vec_model")

# get all the derivations of each word in the search term, and generates a new search term based on these derivations (only if they exist in the dictionary)
# input is the search term to use
# output is a new search term that contains all of the derivations
def getDerivations(searchTerm):
	dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
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

# finds the most similar document to the provided searchTerm using the saved model files
# input requires the following:
# model is a string specifying the model to use. Can be one of "bow", "tfidf", "lsi" or "doc2vec"
# numResults is an int specifying number of similar documents to return
# searchTerm is a (raw) string containing the term to search for
# output is an array containing the index of the similar documents and their similarity value
def search(model, numResults, searchTerm):
	dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
	searchTerm = textPreprocess(searchTerm)
	searchTerm = getDerivations(searchTerm)
	if (searchTerm == []):
		return []
	if model == "bow":
		index = gensim.similarities.SparseMatrixSimilarity.load('./model_files/reports.index')
		index.num_best = numResults

		searchTerm_bow = dictionary.doc2bow(searchTerm)

		similarReports = index[searchTerm_bow]
	elif model == "tfidf":
		tfidf_model = gensim.models.TfidfModel.load('./model_files/reports.tfidf_model')

		tfidf_index = gensim.similarities.SparseMatrixSimilarity.load('./model_files/reports_tfidf.index')
		tfidf_index.num_best = numResults

		searchTerm_bow = dictionary.doc2bow(searchTerm)
		searchTerm_tfidf = tfidf_model[searchTerm_bow]

		similarReports = tfidf_index[searchTerm_tfidf]
	elif model == "lsi":
		tfidf_model = gensim.models.TfidfModel.load('./model_files/reports.tfidf_model')
		lsi_model = gensim.models.LsiModel.load('./model_files/reports.lsi_model')

		lsi_index = gensim.similarities.MatrixSimilarity.load('./model_files/reports_lsi.index')
		lsi_index.num_best = numResults

		searchTerm_bow = dictionary.doc2bow(searchTerm)
		searchTerm_tfidf = tfidf_model[searchTerm_bow]
		searchTerm_lsi = lsi_model[searchTerm_tfidf]

		similarReports = lsi_index[searchTerm_lsi]
	elif model == "lda":
		lda_model = gensim.models.LdaModel.load('./model_files/reports.lda_model')

		lda_index = gensim.similarities.MatrixSimilarity.load('./model_files/reports_lda.index')
		lda_index.num_best = numResults

		searchTerm_bow = dictionary.doc2bow(searchTerm)
		searchTerm_lda = lda_model[searchTerm_bow]

		similarReports = lda_index[searchTerm_lda]
	elif model == "doc2vec":
		model = gensim.models.Doc2Vec.load("./model_files/reports.doc2vec_model")
		# searchTerm_docvec = model.infer_vector(getDerivations(searchTerm))
		searchTerm_docvec = model.infer_vector(searchTerm)
		similarReports = model.docvecs.most_similar([searchTerm_docvec],topn=numResults)
	else:
		return 0

	return similarReports


# simulates a search engine and can be used in testing
# input is the desired model and searchTerm
# no return output but prints the top 5 most similar reports
def searchEngineTest(model, searchTerm):
	print("Search: " + searchTerm)

	reports = getReports()

	similarReports = search(model,5,searchTerm)

	if (similarReports == []):
		print ("ERROR: Invalid search term")

	for reportIdx in similarReports:
		print("----------")
		print("Report #: " + str(reportIdx[0]) + " Similarity: " + str(reportIdx[1]) )
		print(reports[reportIdx[0]])

# performs cross-validation and generates precision-recall curve
# used to compare the accuracy of the searching mechanism of the four models
# input is a string of a filename containing a list of searchTerms to use in the testing
# saves output to files in the directory "./precision_recall/"
def precisionRecall(testFile):
	models = ["bow","tfidf","lsi","lda","doc2vec"]
	tests = []
	with open(testFile,'rb') as file:
		reader = csv.reader(file)
		for row in reader:
			tests.append(row)
	file.close()

	thres = [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5,0.8]

	numReports = [getNumReports(REPORT_FILES[:1]), getNumReports(REPORT_FILES[:2]), getNumReports(REPORT_FILES[:3]), getNumReports()]

	for searchTerm in tests:
		print(searchTerm)
		plt.figure(searchTerm[0])
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title(searchTerm[0])
		with open("precision_recall_rerun/" + searchTerm[0] + ".csv",'w') as writeFile:
			writer = csv.writer(writeFile)

			for model in models:
				writer.writerow([model])
				precision = []
				recall = []
				for i in range(len(thres)):
					truePositive = 0
					retrieved = 0 # retreieved = truePositive + falsePositive
					relevant = 0 # relevant = truePositive + falseNegative

					numResults = getNumReports()
					similarReports = search(model,numResults,searchTerm[0])
					similarReports = [report for report in similarReports if report[1] > thres[i]]
					# pprint(similarReports)

					for reportIdx in similarReports:
						if reportIdx[0] < numReports[0]: # prediction: brains
							if (searchTerm[1] == "Brains"): # actual: brains
								truePositive = truePositive + 1
							# print "brains"
						elif reportIdx[0] < numReports[1]:
							if (searchTerm[1] == "CTPA"):
								truePositive = truePositive + 1
							# print "ctpa"
						elif reportIdx[0] < numReports[2]:
							if (searchTerm[1] == "Plainab"):
								truePositive = truePositive + 1
							# print "plainab"
						elif reportIdx[0] < numReports[3]:
							if (searchTerm[1] == "Pvab"):
								truePositive = truePositive + 1
							# print "pvab"
						else:
							print "error"
					retrieved = retrieved + len(similarReports)
					relevant = relevant + getNumReports(["nlp_data/Cleaned" + searchTerm[1] + "Full.csv"])

					precision.append((truePositive/retrieved) if retrieved else 0)
					recall.append((truePositive/relevant) if relevant else 0)
					writer.writerow([precision[i-1],recall[i-1]])

				writer.writerow("")

				# plot the data point
				plt.plot(recall,precision,label=model)

		writeFile.close()
		plt.legend(loc='lower right')
	plt.show()



# tests the model at classifying reports as either positive or negative based on diagnosis
def labelClassification():
	corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
	corpusList = [list(x) for x in np.asarray(corpus)[:,:,1]]
	reports = getReports()

	numFolds = 5 # number of folds for cross validation

	with open("labelClassification.csv",'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(["score","output label","expected label","report"])

		for j in range(len(REPORT_FILES_LABELLED)):
			writer.writerow("")
			writer.writerow("")
			writer.writerow([DIAGNOSES[j]])

			# initialise figure and plot
			plt.figure(DIAGNOSES[j] + " ROC")
			plt.xlabel("False Positive")
			plt.ylabel("True Positive")
			plt.title(DIAGNOSES[j] + " ROC")

			# fetch corpus and labels
			labelledCorpus = []
			for i in range(getNumReports(REPORT_FILES[:j]),getNumReports(REPORT_FILES[:j])+getNumReports([REPORT_FILES_LABELLED[j]])):
				labelledCorpus.append((corpus[i]))
			labelledCorpus = np.asarray(labelledCorpus)[:,:,1]
			labels = np.asarray(getData([REPORT_FILES_LABELLED[j]]))[:,2]

			############### THIS CODE BLOCK REMOVES THE NUMBER OF NEGATIVE LABELS TO EQUALISE THE DISTRIBUTION OF CLASS LABELS. TO BE REMOVED IN FUTURE.
			count = 0
			deletes = []
			for x in range(len(labels)):
				if (labels[x] == "negative"):
					count = count + 1
					deletes.append(x)
				if (count == (len(labels)-(list(labels).count("positive"))*2)):
					break
			labelledCorpus = np.delete(labelledCorpus,deletes,axis=0)
			labels = np.delete(labels,deletes)
			##################

			# # shuffle the order of data
			# shuffleList = list(zip(labelledCorpus,labels))
			# random.shuffle(shuffleList)
			# labelledCorpus,labels = zip(*shuffleList)
			# labels = np.asarray(labels)
			# labelledCorpus = np.asarray(labelledCorpus)

			numData = len(labels) # size of the labelled data set
			dataPerFold = int(math.ceil(numData/numFolds))


			for n in range(0,numFolds):
				# split training and test data
				train_labelledCorpus,test_labelledCorpus,train_labels,test_labels = train_test_split(labelledCorpus,labels,test_size=0.13)

				# test_labels = labels[n*dataPerFold:min(numData,(n+1)*dataPerFold)]
				# train_labels = np.delete(labels,range(n*dataPerFold,min(numData,(n+1)*dataPerFold)))

				# test_labelledCorpus = labelledCorpus[n*dataPerFold:min(numData,(n+1)*dataPerFold)]
				# train_labelledCorpus = np.delete(labelledCorpus,range(n*dataPerFold,min(numData,(n+1)*dataPerFold)),axis=0)


				# build classifier
				classifier = svm.SVC(kernel='linear').fit(train_labelledCorpus,train_labels)
				# classifier = svm.LinearSVC(C=1.0).fit(train_labelledCorpus,train_labels)
				# classifier = neighbors.KNeighborsClassifier(n_neighbors=3).fit(train_labelledCorpus,train_labels)

				# compute output label and corresponding score
				output_test = classifier.predict(test_labelledCorpus)
				output_train = classifier.predict(train_labelledCorpus)
				output_scores_test = classifier.decision_function(test_labelledCorpus)
				output_scores_train = classifier.decision_function(train_labelledCorpus)

				# sort scores and labels in order
				sortList = list(zip(output_scores_test,output_test,test_labels,test_labelledCorpus))
				sortList.sort()
				output_scores_test,output_test,test_labels,test_labelledCorpus = zip(*sortList)

				# build roc curve and plot
				fp_test,tp_test,_ = roc_curve(test_labels,output_scores_test,pos_label="positive")
				fp_train,tp_train,_ = roc_curve(train_labels,output_scores_train,pos_label="positive")

				plt.plot(fp_test,tp_test,'r',label="train" if n == 0 else "")
				plt.plot(fp_train,tp_train,'b',label="test" if n == 0 else "")
				plt.legend(loc='lower right')


				# save result to file
				for r in range(len(test_labels)):
					reportIdx = corpusList.index(list(test_labelledCorpus[r]))
					writer.writerow("")
					writer.writerow([output_scores_test[r],output_test[r],test_labels[r]])
					writer.writerow([reports[reportIdx]])
		plt.show()
	writeFile.close()



# implements search engine and outputs as according to requirements for front-end integration
def runSearchEngine():
	# process the search term
	if (len(sys.argv) < 2):
		print("ERROR: Please specify an input file")
		sys.exit()
	fileName = str(sys.argv[1])
	fileText = [row.rstrip('\n') for row in open(fileName)]

	if (fileText[1] == "notonlyreturndates"):
		print("Req No.<,>Report Date<,>Report")
	elif (fileText[1] == "onlyreturndates"):
		print("Req No.<,>Report Date")
	else:
		print("ERROR: input file layout error")
		sys.exit()

	data = getData()

	similarReports = search("lsi",50,fileText[0])
	for reportIdx in similarReports:
		year = random.randint(2000,int(fileText[2][0:4])-1)
		month = random.randint(1,12)
		date = random.randint(1,28)
		if (fileText[1] == "notonlyreturndates"):
			print(data[reportIdx[0]][0] + "<,>" + str(year) + str(month).zfill(2) + str(date).zfill(2) + "<,>" + data[reportIdx[0]][1])
		elif (fileText[1] == "onlyreturndates"):
			print(data[reportIdx[0]][0] + "<,>" + str(year) + str(month).zfill(2) + str(date).zfill(2))


if __name__ == '__main__':
	# preprocessReports()
	# buildDictionary()
	buildModels()
	# buildWord2VecModel()
	# buildDoc2VecModel()
	# searchTerm = "chronic small vessel disease"
	# searchTerm = "2400      CT HEAD - PLAIN L3  CT HEAD:  CLINICAL DETAILS:  INVOLVED IN FIGHT, KICKED IN HIS HEAD, VOMITED AFTER THIS WITH EPISODIC STARING EPISODES WITH TEETH GRINDING. ALSO INTOXICATED (BREATH ALCOHOL ONLY 0.06). PROCEDURE:  PLAIN SCANS THROUGH THE BRAIN FROM SKULL BASE TO NEAR VERTEX. IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  REPORT:  VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE AND IT IS SYMMETRICAL AROUND THE MIDLINE.  NORMAL GREY/WHITE DIFFERENTIATION.  NO INTRACEREBRAL HAEMATOMA OR EXTRA AXIAL COLLECTION. NO CRANIAL VAULT FRACTURE SEEN.  COMMENT: STUDY WITHIN NORMAL LIMITS."
	# searchTerm = "GREY/WHITE MATTER DIFFERENTIATION"
	# searchEngineTest("lda",searchTerm)
	# precisionRecall("pr_tests.csv")
	# labelClassification()

	# runSearchEngine()
