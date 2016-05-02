import csv

import sys

import dataUtils
import nltk
import re

# # dataUtils.generateSentencesFromRaw()
# exit(0)
# new = dataUtils.readFromCSVCONVERT('./sentence_label_data/sentences_Brains.csv')
# new = dataUtils.removeDuplicates(new)
# print "converted"
# old = dataUtils.readFromCSV('./sentence_label_data/sentences_ALL.csv')
#
#
#
# REPORT_FILES_BRAINS = ['nlp_data/CleanedBrainsFull.csv']
# REPORT_FILES_CTPA = ['nlp_data/CleanedCTPAFull.csv']
# REPORT_FILES_PLAINAB = ['nlp_data/CleanedPlainabFull.csv']
# REPORT_FILES_PVAB = ['nlp_data/CleanedPvabFull.csv']
#
# hash = dict()
#
# for record in old:
#     hash[record.processedSentence] = record
# print "dict made"
# for record in new:
#     if record.processedSentence in hash:
#         hash[record.processedSentence].diagTag = record.diagTag
#         hash[record.processedSentence].sentTag = record.sentTag
#
# proc = [r for r in old]
#
# dataUtils.writeToCSV('./sentence_label_data/sentences_ALL.csv', proc)

def generateSentenceIDDict():
    sentenceFiles = ["./sentence_label_data/CleanedBrainsFull.csv",
                     "./sentence_label_data/CleanedCTPAFull.csv",
                     "./sentence_label_data/CleanedPlainabFull.csv",
                     "./sentence_label_data/CleanedPvabFull.csv"]

    # Dictionary <sentence, report ID>
    sentIDs = dict()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for fileName in sentenceFiles:
        with open(fileName, 'rb') as file:
            file.readline()  # skip header line
            reader = csv.reader(file)
            for row in reader:
                for sentence in tokenizer.tokenize(row[1]):
                    if sentence not in sentIDs:
                        sentIDs[sentence] = row[0]
                    else:
                        sentIDs[sentence] = sentIDs[sentence] + "," + row[0]

    return sentIDs

def addSentenceIDs(dataFile, sentIDFile='./sentence_label_data/sentence_id_mappings.csv'):
    csv.field_size_limit(sys.maxsize)

    sentIDs = dict(csv.reader(open(sentIDFile, 'rb')))
    labelledData = dataUtils.readFromCSV(dataFile)

    # Set as first report ID corresponding to that sentence
    for row in labelledData:
        row.reportID = re.search("([0-9a-zA-Z]+)", sentIDs[row.sentence]).group(0)

    dataUtils.writeToCSV(dataFile, labelledData)

#write dict to file
# writer = csv.writer(open('./sentence_label_data/sentence_id_mappings.csv', 'wb'))
# for key, value in sentIDs.items():
#     writer.writerow([key, value])