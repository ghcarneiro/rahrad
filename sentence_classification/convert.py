import csv
import sys
import data_utils
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




# addSentenceIDs('./sentence_label_data/sentences_ALL.csv')
# write dict to file
# writer = csv.writer(open('./sentence_label_data/sentence_id_mappings.csv', 'wb'))
# for key, value in sentIDs.items():
#     writer.writerow([key, value])

# data_utils.write_to_csv('./sentence_label_data/sentences_ALL_notags.csv', strip_labels(data_utils.read_from_csv('./sentence_label_data/sentences_ALL.csv')))