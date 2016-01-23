from __future__ import division
import math
import csv
import numpy as np
import gensim
from sklearn import svm
import time
import datetime
import os

import preprocess

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
# builds and saves dictionary and corpus (in BOW form) from report files
def buildDictionary():
	reports = preprocess.getProcessedReports()

	print("files loaded")

	# build dictionary
	dictionary = gensim.corpora.Dictionary(reports)
	# dictionary.filter_extremes(no_below=3)
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
# num_topics tested with 10,20,30,35,40,50
def transform_lda(corpus,dictionary):
	# lda_model = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=30)
	lda_model = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=30)
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
	build_similarityIndex(corpus)

	# transform model using TFIDF
	transform_tfidf(corpus)
	tfidf_corpus = gensim.corpora.MmCorpus('./model_files/reports_tfidf.mm')
	print('Example case report under Tf-Idf transformation: ')
	print(list(tfidf_corpus)[200])

	# transform model using LSI
	transform_lsi(tfidf_corpus,dictionary)
	lsi_corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
	# lsi_model.print_topics()
	print('Example case report under LSI transformation: ')
	print(list(lsi_corpus)[200])

	# # transform model using LDA
	# transform_lda(corpus,dictionary)
	# lda_corpus = gensim.corpora.MmCorpus('./model_files/reports_lda.mm')
	# # lda_model.print_topics()
	# print('Example case report under LDA transformation: ')
	# print(list(lda_corpus)[200])

# function to test the functionality of Word2Vec
def buildWord2VecModel():
	reports = preprocess.getProcessedReports()

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
# doc2vec performs better with dbow than dm
# tested with hidden layer size 100,200,300
def buildDoc2VecModel():
	reports = preprocess.getProcessedReports()

	# construct sentences from reports
	taggedDocuments = []
	for i in range(len(reports)):
		taggedDocument = gensim.models.doc2vec.TaggedDocument(words= reports[i], tags= [i])
		taggedDocuments.append(taggedDocument)


	# model = gensim.models.Doc2Vec(taggedDocuments)
	model = gensim.models.Doc2Vec(size=100, min_count=5, workers=16,dm=1, dbow_words=1,negative=20)

	model.build_vocab(taggedDocuments)

	model.alpha = 0.025 # learning rate

	for epoch in range(10):
		print(epoch)
		model.train(taggedDocuments)
		model.alpha -= 0.001
		model.min_alpha = model.alpha


	model.save("./model_files/reports.doc2vec_model")

# generated a svm model for classifying reports as either positive or negative based on diagnosis
# uses a MmCorpus file
def testClassification():
	threashold = 0.001
	corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
	#convert the corpus to a numpy matrix, take the transpose and convert it to a list
	corpusList = [list(x) for x in zip(*gensim.matutils.corpus2dense(corpus,corpus.num_terms,dtype=np.float64))]
	# corpusList = [list(x) for x in np.asarray(corpus)[:,:,1]]
	reports = preprocess.getReports()

	numFolds = 5 # number of folds for cross validation
	# Create the output directory
	directory = "label_tests/" + datetime.datetime.now().strftime('%m_%d_%H_%M') +"/"
	os.makedirs(directory)
	with open(directory+"labelClassification.csv",'w') as writeFile:
		writer = csv.writer(writeFile)
		writer.writerow(["score","output label","expected label","report"])

		for j in range(len(REPORT_FILES_LABELLED)):
			writer.writerow("")
			writer.writerow("")
			writer.writerow([DIAGNOSES[j]])

			# fetch corpus and labels
			labelledCorpus = []
			unlabelledCorpus = []
			# The labeled data is at the start of the data set
			# Get the ids in the corpus of these first labeled examples for each class
			for i in range(preprocess.getNumReports(REPORT_FILES[:j]),preprocess.getNumReports(REPORT_FILES[:j])+preprocess.getNumReports([REPORT_FILES_LABELLED[j]])):
				labelledCorpus.append(corpusList[i])
			for i in range(preprocess.getNumReports(REPORT_FILES[:j])+preprocess.getNumReports([REPORT_FILES_LABELLED[j]]),preprocess.getNumReports(REPORT_FILES[:j])+preprocess.getNumReports([REPORT_FILES[j]])):
				unlabelledCorpus.append(corpusList[i])
			labels = np.asarray(preprocess.getData([REPORT_FILES_LABELLED[j]]))[:,2]
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

			numData = len(labels) # size of the labelled data set

			# build classifier
			classifier = svm.SVC(kernel='linear').fit(labelledCorpus,labels)

			# compute output label and corresponding score
			output_test = classifier.predict(unlabelledCorpus)
			output_scores_test = classifier.decision_function(unlabelledCorpus)

			# sort scores and labels in order
			sortList = list(zip(output_scores_test,output_test,unlabelledCorpus))
			sortList.sort()
			output_scores_test,output_test,unlabelledCorpus = zip(*sortList)

			# save result to file
			for r in range(len(unlabelledCorpus)):
				if (abs(output_scores_test[r]) < threashold):
					reportIdx = corpusList.index(list(unlabelledCorpus[r]))
					writer.writerow("")
					writer.writerow([reportIdx,output_scores_test[r],output_test[r]])
					writer.writerow([reports[reportIdx]])
	writeFile.close()
