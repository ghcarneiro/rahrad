import csv
import re
import sys

import nltk

sys.path.append("..")  # Adds higher directory to python modules path.

import preprocess


class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.processed_sentence = ""
        self.diag_probs = []
        self.sent_probs = []
        self.diag_tag = ""
        self.sent_tag = ""
        self.report_id = ""


# Write all sentences to file
def write_to_csv(sentence_file, sentence_tags):
    with open(sentence_file, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header 'sentence, labels'
        writer.writerow(
            ["sentence"] + ["processed sentence"] + ["diagnostic label"] + ["sentiment label"] + ["report id"])
        for row in sentence_tags:
            writer.writerow(
                [row.sentence] + [row.processed_sentence] + [row.diag_tag] + [row.sent_tag] + [row.report_id])


# Read in all sentences defined in the given file
def read_from_csv(sentence_file):
    data = []

    with open(sentence_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            tmp = SentenceRecord(row[0])
            tmp.processed_sentence = row[1]
            tmp.diag_tag = row[2]
            tmp.sent_tag = row[3]
            tmp.report_id = row[4]
            data.append(tmp)

    # Return the read objects, but cut off the first row which was headers
    return data[1:]

# Generate the unlabelled sentence data from the given datafile
def generate_sentences(data_files):
    data = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for file_name in data_files:
        with open(file_name, 'rb') as file:
            file.readline()  # skip header line
            reader = csv.reader(file)
            for row in reader:
                for sentence in tokenizer.tokenize(row[1]):
                    tmp = SentenceRecord(sentence)
                    tmp.processed_sentence = " ".join(
                        preprocess.textPreprocess(sentence, removeNegationsFromSentences=False))
                    tmp.report_id = row[0]
                    data.append(tmp)

    return data


def remove_duplicates(data):
    added = set()
    res = []

    for row in data:
        if row.processedSentence not in added:
            res.append(row)
            added.add(row.processedSentence)

    return res


def generate_sentences_from_raw():
    confirm = raw_input("Are you sure you want to regenerate? (yes/no) ")
    if confirm == "yes":

        files = ["./sentence_label_data/CleanedBrainsFull.csv",
                 "./sentence_label_data/CleanedCTPAFull.csv",
                 "./sentence_label_data/CleanedPlainabFull.csv",
                 "./sentence_label_data/CleanedPvabFull.csv"]

        output_file = './sentence_label_data/sentences_ALL.csv'
        write_to_csv(output_file, remove_duplicates(generate_sentences(files)))
    else:
        print "Cancelled."


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
            tmp = SentenceRecord(row[0])
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

    write_to_csv(data_file, labelled_data)
