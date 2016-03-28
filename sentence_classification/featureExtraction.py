import gensim
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


class GensimModel(object):
    def __init__(self, processedSentences, labels):
        if len(processedSentences) != len(labels):
            raise ValueError("Dimension of sentences and labels do not match")

        # Not sure why the number of features corresponds to this length
        self.numFeatures = len(processedSentences)
        self.labels = labels

        self.dictionary = gensim.corpora.Dictionary()

        for sentence in processedSentences:
            self.dictionary.add_documents(sentence.split())

        # Create corpus in the form of word count matrices for current tagged sentences to train with.
        self.corpus = [self.dictionary.doc2bow(sentence.split()) for sentence in processedSentences]

        # Create lsi model from corpus
        self.lsi_model = gensim.models.LsiModel(self.corpus, id2word=self.dictionary)
        self.corpus = self.lsi_model[self.corpus]

        self.corpusList = self.convSparse2Dense(self.corpus)

    def convSparse2Dense(self, sparse):
        return [list(x) for x in zip(*gensim.matutils.corpus2dense(sparse, self.numFeatures))]

    def getFeatures(self, sentence):
        bowSentence = self.dictionary.doc2bow(sentence.split)
        sparseResult = self.lsi_model[bowSentence]
        return self.convSparse2Dense([sparseResult])[0]

class skModel(object):
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.svd = TruncatedSVD(n_components=10, random_state=42)

    # Input must be a list of strings
    # Returns the strings transformed to final representation
    def getFeatures(self, data):
        return self.svd.transform(self.vectorizer.transform(data).toarray())

    # Applies given data to fitting the model
    # Must be a list of strings
    def fit(self, trainingData):
        # Applies given sentences to fitting the vectorizer
        train = self.vectorizer.fit_transform(trainingData).toarray()
        # Passes vectorized sentences to fit SVD
        self.svd.fit(train)
