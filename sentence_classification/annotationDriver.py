import sys, signal

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from dataUtils import *
from featureExtraction import *

import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
        print abs(sentence.diagProbs[0] - sentence.diagProbs[1])
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

def compareDiagProb(item1, item2):
    diff1 = 1 if item1.diagProbs == [] else abs(item1.diagProbs[0] - item1.diagProbs[1])
    diff2 = 1 if item2.diagProbs == [] else abs(item2.diagProbs[0] - item2.diagProbs[1])

    if diff1 == diff2 :
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1

#################
### Labelling ###
#################

if type == "generate" and tag == "raw":
    generateSentencesFromRaw()
    exit(0)
else:
    data = []

    # Allows conversion from old format, ensures is in object format
    if convCSV:
        data = convCSV2Obj(sentenceFile)
    else:
       data = readPickle(pickleFile)

    data = labelSentences(data, tag)

########################
### Trial prediction ###
########################

# Extract just the sentences that are tagged and preprocess them
print "Extracting tagged sentences for classification"
taggedSentences = [x.processedSentence for x in data if x.diagTag != ""]
labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != ""]

print "Building feature extraction model"
model = skModel(taggedSentences, labels)

print "Building classifier"
# Create and fit RandomForest classifier with annotations
forest = RandomForestClassifier()
forest.fit(model.corpus, model.labels)

# Take smaller working set, not need to classify everything
workingList = [x for x in data if x not in taggedSentences]
if len(workingList) > 100:
    workingList = workingList[0:99]

print "Calculating classifications"
for row in workingList:
    row.diagProbs = forest.predict_proba(model.getFeatures(row.processedSentence))[0]

print "Sorting data"
data.sort(cmp=compareDiagProb)

print "Saving data"
writePickle(pickleFile, data)

## TODO
# Think of better solution to medical dictionary filepath