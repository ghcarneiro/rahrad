import csv
import sys

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
        writer.writerow(["sentence"] + ["processed sentence"] + ["diagnostic label"] + ["sentiment label"] + ["report id"])
        for row in sentence_tags:
            writer.writerow([row.sentence] + [row.processed_sentence] + [row.diag_tag] + [row.sent_tag] + [row.report_id])


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


# Converts CSV in the original 3 column format into current version
def read_from_csv_convert(sentenceFile):
    data = []

    with open(sentenceFile, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            tmp = SentenceRecord(row[0])
            tmp.processed_sentence = " ".join(preprocess.textPreprocess(row[0], removeNegationsFromSentences=False))
            tmp.diag_tag = row[1]
            tmp.sent_tag = row[2]
            data.append(tmp)

    # Return the read objects, but cut off the first row which was headers
    return data[1:]


# TODO rewrite to incorporate report IDs
# Generate the unlabelled sentence data from the given datafile
def generate_sentences(dataFiles):
    sentence_tags = []
    sentences = preprocess.getAllSentences(dataFiles)

    for sentence in sentences:
        tmp = SentenceRecord(sentence)
        tmp.processed_sentence = " ".join(preprocess.textPreprocess(sentence, removeNegationsFromSentences=False))
        sentence_tags.append(tmp)

    return sentence_tags


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
        types = ["Brains", "CTPA", "Plainab", "Pvab"]
        files = []
        for type in types:
            files.append("../nlp_data/Cleaned" + type + "Full.csv")

        output_file = './sentence_label_data/sentences_ALL.csv'
        write_to_csv(output_file, remove_duplicates(generate_sentences(files)))
    else:
        print "Cancelled."
