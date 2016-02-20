import gensim
import preprocess
import rnn
# finds the most similar document to the provided searchTerm using the saved model files
# input requires the following:
# model is a string specifying the model to use. Can be one of "bow", "tfidf", "lsi" or "doc2vec"
# numResults is an int specifying number of similar documents to return
# searchTerm is a (raw) string containing the term to search for
# output is an array containing the index of the similar documents and their similarity value
def search(model, numResults, searchTerm):
	dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
	origSearchTerm = searchTerm
	searchTerm = preprocess.textPreprocess(searchTerm)
	# searchTerm = preprocess.getDerivations(searchTerm)
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

		searchTerm_docvec = model.infer_vector(searchTerm)

		similarReports = model.docvecs.most_similar([searchTerm_docvec],topn=numResults)
	elif model == "rnn":
		searchTerm_rnn = rnn.getReportSearchTerm(origSearchTerm)

		similarReports = rnn.most_similar_reports(searchTerm_rnn,topn=numResults)
	else:
		return 0

	return similarReports


# simulates a search engine and can be used in testing
# input is the desired model and searchTerm
# no return output but prints the top 5 most similar reports
def searchEngineTest(model, searchTerm):
	print("Search: " + searchTerm)

	reports = preprocess.getReports()
	similarReports = search(model,5,searchTerm)

	if (similarReports == []):
		print ("ERROR: Invalid search term")

	for reportIdx in similarReports:
		print("----------")
		print("Report #: " + str(reportIdx[0]) + " Similarity: " + str(reportIdx[1]) )
		print(reports[reportIdx[0]])
