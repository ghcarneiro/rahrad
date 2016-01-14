import gensim

# builds and saves dictionary and corpus (in BOW form) from report files
def buildDictionary():
	reports = getProcessedReports()

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
# doc2vec performs better with dbow than dm
# tested with hidden layer size 100,200,300
def buildDoc2VecModel():
	reports = getProcessedReports()

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
