import csv
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import preprocess

class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.processedSentence = []
        self.diagProbs = []
        self.sentProbs = []
        self.diagTag = ""
        self.sentTag = ""


# Write all sentences to file
def writeToCSV(sentenceFile, sentenceTags):
    with open(sentenceFile, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header 'sentence, labels'
        writer.writerow(["sentence"] + ["processed sentence"] + ["diagnostic label"] + ["sentiment label"])
        for row in sentenceTags:
            writer.writerow([row.sentence] + [row.processedSentence] + [row.diagTag] + [row.sentTag])


# Read in all sentences defined in the given file
def readFromCSV(sentenceFile):
    data = []

    with open(sentenceFile, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            tmp = SentenceRecord(row[0])
            tmp.processedSentence = row[1]
            tmp.diagTag = row[2]
            tmp.sentTag = row[3]
            data.append(tmp)

    # Return the read objects, but cut off the first row which was headers
    return data[1:]


# Generate the unlabelled sentence data from the given datafile
def generateSentences(dataFiles):
    sentenceTags = []
    sentences = preprocess.getAllSentences(dataFiles)

    for sentence in sentences:
        tmp = SentenceRecord(sentence)
        tmp.processedSentence = " ".join(preprocess.textPreprocess(sentence, removeNegationsFromSentences=True))
        sentenceTags.append(tmp)

    return sentenceTags


def removeDuplicates(data):
    added = set()
    res = []

    for row in data:
        if row.processedSentence not in added:
            res.append(row)
            added.add(row.processedSentence)

    return res


def generateSentencesFromRaw():
    confirm = raw_input("Are you sure you want to regenerate? (yes/no) ")
    if confirm == "yes":
        types = ["Brains", "CTPA", "Plainab", "Pvab"]
        files = []
        for type in types:
            files.append("../nlp_data/Cleaned" + type + "Full.csv")

        outputFile = './sentence_label_data/sentences_ALL.csv'
        writeToCSV(outputFile, removeDuplicates(generateSentences(files)))
    else:
        print "Cancelled."
