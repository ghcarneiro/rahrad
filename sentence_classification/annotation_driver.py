import sys, signal
from random import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_utils import *
import pipelines


# Stop the process being killed accidentally so results aren't lost.
def signal_handler(signal, frame):
    print "To exit please press CTL+D so the results can be saved."

signal.signal(signal.SIGINT, signal_handler)

usage = "USAGE: " + sys.argv[0] + " (diagnostic|sentiment)"

if len(sys.argv) != 2:
    print usage
    sys.exit(1)

tag = sys.argv[1]

dataFile = './sentence_label_data/sentences_ALL_notags.csv'


if tag == "diagnostic":
    diffTolerance = 0.15
elif tag == "sentiment":
    diffTolerance = 0.5


# Control loop for labelling sentences, returns the data list with tags changed
def labelSentences(sentenceTags, tag, ignoreProbs):
    userExits = False
    diff = 0

    if not isinstance(sentenceTags[0], SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    for sentence in sentenceTags:
        if tag == "diagnostic" and sentence.diagTag == "" and len(sentence.diagProbs) != 0 or ignoreProbs:
            if not ignoreProbs:
                diff = abs(sentence.diagProbs[0] - sentence.diagProbs[1])
            userExits, ans = getTags(
                "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                ["p", "n", "u"],
                sentence)
            print ""

            sentence.diagTag = ans

        elif tag == "sentiment" and sentence.diagTag == "p" and sentence.sentTag == "" and len(sentence.sentProbs) != 0 or ignoreProbs:
            if not ignoreProbs:
                diff = abs(sentence.sentProbs[0] - sentence.sentProbs[1])
            userExits, ans = getTags(
                "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                ["p", "n", "u"],
                sentence)
            print ""

            sentence.sentTag = ans

        if not ignoreProbs and diff > diffTolerance:
            print "Uncertainty tolerance reached. Rerun to classify more."
            userExits = True

        if userExits:
            break

    if (userExits):
        print "Exiting: Saving progress so far, come back and tag more later."
    else:
        print "Finished tagging! If you were tagging diagnostic, " \
              "change 'diagnostic' to 'sentiment' to tag the second feature."

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
            if ans == "quit":
                userExits = True
                break
            else:
                if ans in possibleAnswers:
                    return userExits, ans
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

    if diff1 == diff2:
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1


def compareSentProb(item1, item2):
    diff1 = 1 if item1.sentProbs == [] else abs(item1.sentProbs[0] - item1.sentProbs[1])
    diff2 = 1 if item2.sentProbs == [] else abs(item2.sentProbs[0] - item2.sentProbs[1])

    if diff1 == diff2:
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1


# Generates the list of SentenceRecords from original data.
# WILL DESTROY EXISTING TAGS
if tag == "generate":
    generateSentencesFromRaw()
    exit(0)

########################
### Data Retrieval   ###
########################

print "Reading data file"
data = readFromCSV(dataFile)

########################
### Classification   ###
########################

taggedSentences = []
labels = []

ignoreProbs = False

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
    print "Number of tagged sentences: " + str(len(taggedSentences))
    print "Building feature extraction model and classifier"
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(taggedSentences, labels)

    # Take smaller working set, not need to classify everything
    workingList = [x for x in data if x not in taggedSentences]
    if len(workingList) > 300:
        workingList = workingList[0:299]

    print "Calculating classifications"
    if tag == "diagnostic":
        for row in workingList:
            row.diagProbs = pipe.predict_proba([row.processedSentence])[0]
            # print row.diagProbs
    elif tag == "sentiment":
        for row in workingList:
            row.sentProbs = pipe.predict_proba([row.processedSentence])

    print "Sorting data"
    if tag == "diagnostic":
        data.sort(cmp=compareDiagProb)
    elif tag == "sentiment":
        data.sort(cmp=compareSentProb)
else:
    ignoreProbs = True
    print "Classification skipped, there are no tagged sentences"

#################
### Labelling ###
#################
data = labelSentences(data, tag, ignoreProbs)

print "Saving data"
shuffle(data)
writeToCSV(dataFile, data)