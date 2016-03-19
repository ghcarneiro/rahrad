from six import string_types

import preprocess, sys, csv, signal
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import gensim
import pdb

# Stop the process being killed accidentally so results aren't lost.
def signal_handler(signal, frame):
    print "To exit please press CTL+D so the results can be saved."

signal.signal(signal.SIGINT, signal_handler)

usage = "USAGE: " + sys.argv[0] + " (Brains|CTPA|Plainab|Pvab) (diagnostic|sentiment)"

if len(sys.argv) != 3:
    print usage
    sys.exit(1)

type = sys.argv[1]
tag = sys.argv[2]

dataFile = "./nlp_data/Cleaned" + type + "Full.csv"
sentenceFile = './sentence_label_data/sentences_' + type + '.csv'
corpusFile = './sentence_label_data/corpus_' + type + '.csv'

# sentences with tags will be represented as lists.
# 0 contains the sentence
# 1 contains the diagnostic tag
# 2 contains the sentiment tag

# Generate the unlabelled sentence data from the given datafile
def generateSentences(dataFile):
    sentenceTags = []
    sentences = preprocess.getSentences([dataFile])

    for sentence in sentences:
        sentenceTags.append([sentence, "", ""])

    return sentenceTags

# Write all sentences to file
def writeToCSV(sentenceFile, sentenceTags):
    with open(sentenceFile, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header 'sentence, labels'
        writer.writerow(["sentence"] + ["diagnostic label"] + ["sentiment label"])
        writer.writerows(sentenceTags)

# Read in all sentences defined in the given file
def readFromCSV(sentenceFile):
    sentenceTags = []

    with open(sentenceFile,'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            sentenceTags.append(row)

    # Return the read pairs, but cut off the first row which was headers
    return sentenceTags[1:]

def labelSentences(sentenceTags, tag):
    userExits = False
    for sentence in sentenceTags:
        # If this pair has not been labelled.
        if tag == "diagnostic":
            if sentence[1] is "":
                userExits = getTags(
                    "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                    1,
                    ["p", "n", "u"],
                    sentence)
                print ""
        elif tag == "sentiment":
            if sentence[1] is not "" and sentence[2] is "":
                userExits = getTags(
                    "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                    2,
                    ["p", "n", "u"],
                    sentence)
                print ""
        else:
            print "Tag not recognised"
            sys.exit(1)

        if userExits:
            break

    if(userExits):
        print "Exiting: Saving progress so far, come back and tag more later."
    else:
        print "Finished tagging! If you were tagging diagnostic, change 'diagnostic' to 'sentiment' to tag the second feature."

    return sentenceTags

def getTags(prompt, index, possibleAnswers, row):
    userExits = False

    print "---> " + row[0]
    print prompt

    while True:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()
        if ans:
            ans = ans.rstrip()
            if ans in possibleAnswers:
                row[index] = ans
                break
            else:
                print "Invalid input. Valid answers: [" + ", ".join(possibleAnswers) + "]"
                continue
        # This block is entered if the user presses CTL+D
        else:
            userExits = True
            break

    return userExits

def generateSentencesFromRaw():
    confirm = raw_input("Are you sure you want to regenerate? (yes/no) ")
    if confirm == "yes":
        # types = ["Brains", "CTPA", "Plainab", "Pvab"]
        types = ["Plainab"]
        for type in types:
            dataFile = "./nlp_data/Cleaned" + type + "Full.csv"
            sentenceFile = './sentence_label_data/sentences_' + type + '.csv'

            writeToCSV(sentenceFile, generateSentences(dataFile))
    else:
        print "Cancelled."



# if type == "generate" and tag == "raw":
#     generateSentencesFromRaw()
# else:
#     writeToCSV(sentenceFile, labelSentences(readFromCSV(sentenceFile), tag))



#################
### Labelling ###
#################


########################
### Trial prediction ###
########################

# Read in sentences from file
sentences = readFromCSV(sentenceFile)

# Extract just the sen
taggedSentences = [x[0] for x in sentences if x[1] != ""]
labels = [np.float32(x[1] == "p") for x in sentences if x[1] != ""]

processedSentences = [preprocess.textPreprocess(x) for x in taggedSentences]
print len(processedSentences)

# print [x[0] for x in sentences]
# Create dictionary from all sentences to account for future predictions.
dictionary = gensim.corpora.Dictionary(processedSentences)

# Create corpus in the form of word count matrices for current tagged sentences to train with.
corpus = [dictionary.doc2bow(sentence) for sentence in processedSentences]
# gensim.corpora.MmCorpus.serialize(corpusFile, corpus)
# corpus = gensim.corpora.MmCorpus(corpusFile)

# corpus = gensim.corpora.MmCorpus(corpus)
# TFIDF

# tfidf_model = gensim.models.TfidfModel(corpus)
# newCorpus = tfidf_model[corpus]

lsi_model = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=300)
print(lsi_model)
corpus = lsi_model[corpus]

# for x in gensim.matutils.corpus2dense(corpus, len(processedSentences)):
#     print x

def convSparse2Dense(sparse, num_terms):
    return [list(x) for x in zip(*gensim.matutils.corpus2dense(sparse, num_terms))]

corpusList = convSparse2Dense(corpus, len(processedSentences))
# print len(corpusList)
# print labels




# gensim.corpora.MmCorpus.serialize(corpusFile, corpus)
# corpus = gensim.corpora.MmCorpus(corpusFile)

# print(corpus)
# print gensim.matutils.corpus2dense(corpus, corpus.num_terms, dtype=np.float64)
#
# for doc in corpus:
#     print doc

# corpusList = [list(x) for x in zip(*gensim.matutils.corpus2dense(corpus, corpus.num_terms, dtype=np.float64))]

# print corpusList

# No idea if index is needed
# index = gensim.similarities.SparseMatrixSimilarity(newCorpus,num_features=corpus.num_terms)

# print processedSentences[0]
# print dictionary.doc2bow(processedSentences[0])
# print lsi_model[dictionary.doc2bow(processedSentences[0])]


# Use count vectorizer to get numerical representation of sentences (Just for test)
# vectorizer = CountVectorizer(min_df=1)
# train = vectorizer.fit_transform(taggedSentences).toarray()

# # Create and fit RandomForest classifier with annotations
forest = RandomForestClassifier()
forest.fit(corpusList, labels)
#
# for x in lsi_model[dictionary.doc2bow(preprocess.textPreprocess(sentences[0][0]))]:
#     print x
#
# print convSparse2Dense([lsi_model[dictionary.doc2bow(preprocess.textPreprocess(sentences[0][0]))]], 38)[0]


# Run test to see if prediction works on seen data (should be 1)
print "[ "+ str(labels[0]) + " ] -> " + sentences[0][0]
# test = lsi_model(dictionary.doc2bow([sentences[0][0]]))
test = convSparse2Dense([lsi_model[dictionary.doc2bow(preprocess.textPreprocess(sentences[0][0]))]], 40)[0]
print forest.predict_proba(test)
print ""
#
# # Run test to see if prediction works on seen data (should be 1)
# print "[ "+ str(labels[15]) + " ] -> " + sentences[15][0]
# test = vectorizer.transform([sentences[15][0]]).toarray()
# print forest.predict_proba(test)
# print ""
#
# # Run test on unseen data
# print "[ ? ] -> " + sentences[56][0]
# test = vectorizer.transform([sentences[56][0]]).toarray()
# print forest.predict_proba(test)
# print ""
#
# # Run test on unseen data
# print "[ ? ] -> " + sentences[66][0]
# test = vectorizer.transform([sentences[66][0]]).toarray()
# print forest.predict_proba(test)
# print ""
