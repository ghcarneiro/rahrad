import sys, numpy as np
from dataUtils import *
import pipelines
from random import shuffle

# Will only tag if the confidence is above these values
POS_THRESHOLD = 0.95
NEG_THRESHOLD = 0.95

usage = "USAGE: " + sys.argv[0] + " type passes inputFile outputFile"
if len(sys.argv) != 5:
    print usage
    sys.exit(1)

type = sys.argv[1]
passes = int(float(sys.argv[2]))
inputFile = sys.argv[3]
outputFile = sys.argv[4]

if type != "diagnostic" and type != "sentiment":
    raise ValueError("Unknown tag: " + type)

data = readFromCSV(inputFile)


newPos = 0
newNeg = 0

# Each pass contains: classifier training, label prediction, finding the max and placing the respective labels
for i in xrange(passes):
    print ""
    print "PASS " + str(i)

    sentences = []
    labels = []

    if type == "diagnostic":
        sentences = [x.processedSentence for x in data if x.diagTag != "" and x.diagTag != "u"]
        labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != "" and x.diagTag != "u"]
    elif type == "sentiment":
        sentences = [x.processedSentence for x in data if x.sentTag != "" and x.sentTag != "u"]
        labels = [np.float32(x.sentTag == "p") for x in data if x.sentTag != "" and x.sentTag != "u"]

    print "There are " + str(len(sentences)) + " tagged sentences in this pass"

    # Train model and classifier
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(sentences, labels)

    bestPos = 0
    bestNeg = 0
    newPosThisPass = 0
    newNegThisPass = 0

    # Makes predicitions for the unlabelled data in the first 1000 items,
    # also keeps track of the max confidence of each classification
    if type == "diagnostic":
        for item in [item for item in data[:500] if item.diagTag == ""]:
            item.diagProbs = pipe.predict_proba([item.processedSentence])[0]
            bestNeg = max(item.diagProbs[0], bestNeg)
            bestPos = max(item.diagProbs[1], bestPos)
    elif type == "sentiment":
        for item in [item for item in data[:500] if item.sentTag == ""]:
            item.sentProbs = pipe.predict_proba([item.processedSentence])[0]
            bestNeg = max(item.sentProbs[0], bestNeg)
            bestPos = max(item.sentProbs[1], bestPos)

    print "Best positive confidence: " + str(bestPos)
    print "Best negative confidence: " + str(bestNeg)

    # Allocates the labels for each of the determined max values if they lie above the threshold
    if type == "diagnostic":
        for item in [item for item in data[:500] if item.diagTag == ""]:
            if POS_THRESHOLD < bestPos == item.diagProbs[1]:
                item.diagTag = "p"
                newPosThisPass += 1
            if NEG_THRESHOLD < bestNeg == item.diagProbs[0]:
                item.diagTag = "n"
                newNegThisPass += 1
    elif type == "sentiment":
        for item in [item for item in data[:500] if item.sentTag == ""]:
            if POS_THRESHOLD < bestPos == item.sentProbs[1]:
                item.sentTag = "p"
                newPosThisPass += 1
            if NEG_THRESHOLD < bestNeg == item.sentProbs[0]:
                item.sentTag = "n"
                newNegThisPass += 1

    # Shuffle the data to ensure a new set of 1000 is served up on the next pass
    shuffle(data)

    newPos += newPosThisPass
    newNeg += newNegThisPass

    print str(newPosThisPass) + " new positive tags added this pass"
    print str(newNegThisPass) + " new negative tags added this pass"

# Save the ouput of the automatic learning
writeToCSV(outputFile, data)

print ""
print "Completed " + str(passes) + " passes over the data."
print str(newPos) + " new positive tags added"
print str(newNeg) + " new negative tags added"
