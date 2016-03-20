import csv
import pickle
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import preprocess


class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.processedSentence = []
        self.diagProbs = []
        self.sentProbs = []
        self.diagTag = ""
        self.sentTag = ""


def readPickle(filename):
    with open(filename, "rb") as fin:
        return pickle.load(fin)

def writePickle(filename, obj):
    with open(filename, "wb") as fout:
        return pickle.dump(obj, fout)

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

# Generate the unlabelled sentence data from the given datafile
def generateSentences(dataFile):
    sentenceTags = []
    sentences = preprocess.getSentences([dataFile])

    for sentence in sentences:
        tmp = SentenceRecord(sentence)
        tmp.processedSentence = preprocess.textPreprocess(sentence)
        sentenceTags.append(tmp)

    return sentenceTags

def generateSentencesFromRaw():
    confirm = raw_input("Are you sure you want to regenerate? (yes/no) ")
    if confirm == "yes":
        # types = ["Brains", "CTPA", "Plainab", "Pvab"]
        types = ["CTPA"]
        for type in types:
            dataFile = "./nlp_data/Cleaned" + type + "Full.csv"
            pickleFile = './sentence_label_data/sentences_' + type + '_pickle.pk'

            writePickle(pickleFile, generateSentences(dataFile))
    else:
        print "Cancelled."

def convCSV2Obj(sentenceFile):
    sentences = readFromCSV(sentenceFile)
    objs = []

    for sentenceRow in sentences:
        tmp = SentenceRecord(sentenceRow[0])
        tmp.processedSentence = preprocess.textPreprocess(sentenceRow[0])
        tmp.diagTag = sentenceRow[1]
        tmp.sentTag = sentenceRow[2]
        objs.append(tmp)

    return objs