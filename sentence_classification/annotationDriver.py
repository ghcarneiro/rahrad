import sys, signal
from dataUtils import *

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import gensim

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
pickleFile = './sentence_label_data/sentences_' + type + '_pickle.pk'
corpusFile = './sentence_label_data/corpus_' + type + '.mm'

convCSV = False

def labelSentences(sentenceTags, tag):
    userExits = False

    if not isinstance(sentenceTags[0], SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    for sentence in sentenceTags:
        # If this pair has not been labelled.
        if tag == "diagnostic":
            if sentence.diagTag is "":
                userExits, ans = getTags(
                    "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                    ["p", "n", "u"],
                    sentence)
                print ""

                sentence.diagTag = ans

        elif tag == "sentiment":
            if sentence.diagTag is not "" and sentence.sentTag is "":
                userExits, ans = getTags(
                    "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                    ["p", "n", "u"],
                    sentence)
                print ""

                sentence.sentTag = ans
        else:
            print "Tag not recognised: " + tag
            sys.exit(1)

        if userExits:
            break

    if(userExits):
        print "Exiting: Saving progress so far, come back and tag more later."
    else:
        print "Finished tagging! If you were tagging diagnostic, change 'diagnostic' to 'sentiment' to tag the second feature."

    return sentenceTags

def getTags(prompt, possibleAnswers, row):
    userExits = False

    if not isinstance(row, SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    print "---> " + row.sentence
    print prompt

    while True:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()
        if ans:
            ans = ans.rstrip()
            if ans in possibleAnswers:
                return userExits, ans
                break
            else:
                print "Invalid input. Valid answers: [" + ", ".join(possibleAnswers) + "]"
                continue
        # This block is entered if the user presses CTL+D
        else:
            userExits = True
            break

    return userExits, ""



#################
### Labelling ###
#################


if type == "generate" and tag == "raw":
    generateSentencesFromRaw()
else:
    data = []

    # Allows conversion from old format, ensures is in object format
    if convCSV:
        data = convCSV2Obj(sentenceFile)
    else:
       data = readPickle(pickleFile)

    for obj in data[1:5]:
        print obj.sentence
        print obj.processedSentence
        print obj.diagTag
        print obj.sentTag
    writePickle(pickleFile, labelSentences(data, tag))

########################
### Trial prediction ###
########################
#
#
# class Model(object):
#     def __init__(self, tokenisedSentences, labels):
#         if len(tokenisedSentences) != len(labels):
#             raise ValueError("Dimension of sentences and labels do not match")
#
#         # Not sure why the number of features corresponds to this length
#         self.numFeatures = len(tokenisedSentences)
#         self.labels = labels
#
#         self.dictionary = gensim.corpora.Dictionary(tokenisedSentences)
#         # Create corpus in the form of word count matrices for current tagged sentences to train with.
#         self.corpus = [self.dictionary.doc2bow(sentence) for sentence in tokenisedSentences]
#
#         # Create lsi model from corpus
#         self.lsi_model = gensim.models.LsiModel(self.corpus, id2word=self.dictionary)
#         self.corpus = self.lsi_model[self.corpus]
#
#         self.corpusList = self.convSparse2Dense(self.corpus)
#
#     def convSparse2Dense(self, sparse):
#         return [list(x) for x in zip(*gensim.matutils.corpus2dense(sparse, self.numFeatures))]
#
#     def getFeatures(self, sentence):
#         processedSentence = preprocess.textPreprocess(sentence)
#         bowSentence = self.dictionary.doc2bow(processedSentence)
#         sparseResult = self.lsi_model[bowSentence]
#         return self.convSparse2Dense([sparseResult])[0]
#
# # Read in sentences from file
# sentences = readFromCSV(sentenceFile)
# #
# # # Extract just the sentences that are tagged and preprocess them
# taggedSentences = [x[0] for x in sentences if x[1] != ""]
# labels = [np.float32(x[1] == "p") for x in sentences if x[1] != ""]
#
# ## I think it might be worth only doing this once as it takes a long time!
# processedSentences = [preprocess.textPreprocess(x) for x in taggedSentences]
#
# model = Model(processedSentences, labels)
#
# # # Create and fit RandomForest classifier with annotations
# forest = RandomForestClassifier()
# forest.fit(model.corpusList, model.labels)
#
# for i, sentences in enumerate(taggedSentences):
#     print "[ " + str(labels[i]) + " ] -> " + str(forest.predict([model.getFeatures(taggedSentences[i])]))
