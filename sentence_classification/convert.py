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

def strip_labels(data):
    for row in data:
        row.diag_tag = ""
        row.sent_tag = ""

    return data


def generate_sentence_id_dict():
    sentence_files = ["./sentence_label_data/CleanedBrainsFull.csv",
                     "./sentence_label_data/CleanedCTPAFull.csv",
                     "./sentence_label_data/CleanedPlainabFull.csv",
                     "./sentence_label_data/CleanedPvabFull.csv"]

    # Dictionary <sentence, report ID>
    sentence_ids = dict()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for file_name in sentence_files:
        with open(file_name, 'rb') as file:
            file.readline()  # skip header line
            reader = csv.reader(file)
            for row in reader:
                for sentence in tokenizer.tokenize(row[1]):
                    if sentence not in sentence_ids:
                        sentence_ids[sentence] = row[0]
                    else:
                        sentence_ids[sentence] = sentence_ids[sentence] + "," + row[0]

    return sentence_ids


def add_sentence_report_ids(data_file, sentence_id_file='./sentence_label_data/sentence_id_mappings.csv'):
    csv.field_size_limit(sys.maxsize)

    sentence_ids = dict(csv.reader(open(sentence_id_file, 'rb')))
    labelled_data = []

    with open(data_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            tmp = data_utils.SentenceRecord(row[0])
            tmp.processed_sentence = row[1]
            tmp.diag_tag = row[2]
            tmp.sent_tag = row[3]
            labelled_data.append(tmp)

    # Return the read objects, but cut off the first row which was headers
    labelled_data = labelled_data[1:]

    # Set as first report ID corresponding to that sentence
    for row in labelled_data:
        query = row.sentence.replace('\\n', '')
        query = query.replace('\n', '')
        row.report_id = re.search("([0-9a-zA-Z]+)", sentence_ids[query]).group(0)

    data_utils.write_to_csv(data_file, labelled_data)


# addSentenceIDs('./sentence_label_data/sentences_ALL.csv')
# write dict to file
# writer = csv.writer(open('./sentence_label_data/sentence_id_mappings.csv', 'wb'))
# for key, value in sentIDs.items():
#     writer.writerow([key, value])

# data_utils.write_to_csv('./sentence_label_data/sentences_ALL_notags.csv', strip_labels(data_utils.read_from_csv('./sentence_label_data/sentences_ALL.csv')))