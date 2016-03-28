import csv
import cPickle as pickle
import sys

from nltk.corpus import stopwords

sys.path.append("..") # Adds higher directory to python modules path.

import preprocess

#load set of stop words
stop = set(stopwords.words("english"))

#load dictionary of specialist lexicon
file = open('../dictionary_files/medical.pkl', 'r')
medical = pickle.load(file)
file.close()


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
        for row in sentenceTags:
            writer.writerow([row.sentence] + [row.diagTag] + [row.sentTag])

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
    sentences = preprocess.getAllSentences([dataFile])

    for sentence in sentences:
        tmp = SentenceRecord(sentence)
        tmp.processedSentence = " ".join(preprocess.textPreprocess(sentence, stop=stop, medical=medical))
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
        for type in types:
            dataFile = "../nlp_data/Cleaned" + type + "Full.csv"
            pickleFile = './sentence_label_data/sentences_' + type + '_pickle.pk'

            writePickle(pickleFile, removeDuplicates(generateSentences(dataFile)))
    else:
        print "Cancelled."

# Sentences must first be generated
def mapFromOld(oldData, newData):
    dictionary = dict()
    # Add all records into a dictionary
    for row in newData:
        dictionary[row.sentence] = row

    # Go through sentences that were tagged and find their corresponding sentence
    # Transfer tags to this entry
    for row in oldData:
        if row.sentence in dictionary:
            print "found"
            current = dictionary[row.sentence]  # Sentence in new data
            current.diagTag = row.diagTag
            current.sentTag = row.sentTag
        else:
            print "skipped"


def convCSV2Obj(sentenceFile):
    sentences = readFromCSV(sentenceFile)
    objs = []

    for sentenceRow in sentences:
        tmp = SentenceRecord(sentenceRow[0])
        tmp.processedSentence = preprocess.textPreprocess(sentenceRow[0], stop=stop, medical=medical)
        tmp.diagTag = sentenceRow[1]
        tmp.sentTag = sentenceRow[2]
        objs.append(tmp)

    return objs