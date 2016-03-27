import sys, signal
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataUtils import *
from featureExtraction import *


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

sentenceFile = './sentence_label_data/sentences_' + type + '.csv'
pickleFile = './sentence_label_data/sentences_' + type + '_pickle.pk'

# Flag to select converting from old csv format
convCSV = False

# Control loop for labelling sentences, returns the data list with tags changed
def labelSentences(sentenceTags, tag):
    userExits = False

    if not isinstance(sentenceTags[0], SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    for sentence in sentenceTags:
        # If this pair has not been labelled.
        if tag == "diagnostic":
            if sentence.diagTag is "":
                print model.getFeatures(sentence.processedSentence)
                ## REMOVE THIS LATER
                print abs(sentence.diagProbs[0] - sentence.diagProbs[1])
                userExits, ans = getTags(
                    "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                    ["p", "n", "u"],
                    sentence)
                print ""

                sentence.diagTag = ans
                if sentence.diagProbs == []:
                    print "Rerun program to learn from the recent tags"
                    userExits = True

        elif tag == "sentiment":
            if sentence.diagTag is not "" and sentence.sentTag is "":
                # Remove this later
                print abs(sentence.sentProbs[0] - sentence.sentProbs[1])
                userExits, ans = getTags(
                    "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                    ["p", "n", "u"],
                    sentence)
                print ""

                sentence.sentTag = ans
                if sentence.sentProbs == []:
                    print "Rerun program to learn from the recent tags"
                    userExits = True

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

# Controls tagging of single sentence
# Returns whether the user wants to exit and the tag chosen
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

def compareSentProb(item1, item2):
    diff1 = 1 if item1.sentProbs == [] else abs(item1.sentProbs[0] - item1.sentProbs[1])
    diff2 = 1 if item2.sentProbs == [] else abs(item2.sentProbs[0] - item2.sentProbs[1])

    if diff1 == diff2 :
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1

# Generates the list of SentenceRecords from original data.
# WILL DESTROY EXISTING TAGS
if type == "generate" and tag == "raw":
    generateSentencesFromRaw()
    exit(0)

########################
### Data Retrieval   ###
########################

data = []

# Allows conversion from old format, ensures is in object format
if convCSV:
    data = convCSV2Obj(sentenceFile)
else:
   data = readPickle(pickleFile)

########################
### Classification   ###
########################

taggedSentences = []
labels = []

# Extract just the sentences that are tagged and were not unsure and preprocess them
print "Extracting tagged sentences for classification"

if tag == "diagnostic":
    taggedSentences = [x.processedSentence for x in data if x.diagTag != "" and x.diagTag != "u"]
    labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != "" and x.diagTag != "u"]
elif tag == "sentiment":
    taggedSentences = [x.processedSentence for x in data if x.sentTag != "" and x.sentTag != "u"]
    labels = [np.float32(x.sentTag == "p") for x in data if x.sentTag != "" and x.sentTag != "u"]
else:
    raise ValueError("Unknown tag: " + tag)

if len(taggedSentences) != 0:
    print "Building feature extraction model"
    model = skModel(taggedSentences)

    print "Building classifier"
    # Create and fit RandomForest classifier with annotations
    forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=3)
    forest.fit(model.corpus, labels)

    # Take smaller working set, not need to classify everything
    workingList = [x for x in data if x not in taggedSentences]
    if len(workingList) > 300:
        workingList = workingList[0:299]


    print "Calculating classifications"
    if tag == "diagnostic":
        for row in workingList:
            row.diagProbs = forest.predict_proba(model.getFeatures(row.processedSentence))[0]
    elif tag == "sentiment":
        for row in workingList:
            row.sentProbs = forest.predict_proba(model.getFeatures(row.processedSentence))[0]

    print "Sorting data"
    if tag == "diagnostic":
        data.sort(cmp=compareDiagProb)
    elif tag == "sentiment":
        data.sort(cmp=compareSentProb)
else:
    print "Classification skipped, there are no tagged sentences"

#################
### Labelling ###
#################

procSent = [x.processedSentence for x in data]
setProcSent = set(procSent)

print len(procSent)
print len(setProcSent)

# data = labelSentences(data, tag)


print "Saving data"
writePickle(pickleFile, data)
writeToCSV(sentenceFile, data)

## TODO
# Think of better solution to medical dictionary filepath
# Include the sentiment portion in active learning