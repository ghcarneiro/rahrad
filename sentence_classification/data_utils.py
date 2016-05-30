import csv
import random
import sys
import matplotlib.pyplot as plt
import nltk
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

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
        self.report_class = -1
        self.feature_vector = []


# Write all sentences to file
def write_to_csv(sentence_file, sentence_tags):
    with open(sentence_file, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header 'sentence, labels'
        writer.writerow(
            ["sentence"] + ["processed sentence"] + ["diagnostic label"] + ["sentiment label"] + ["report id"] + [
                "report_class"])
        for row in sentence_tags:
            writer.writerow(
                [row.sentence] + [row.processed_sentence] + [row.diag_tag] + [row.sent_tag] + [row.report_id] + [
                    row.report_class])


# Read in all sentences defined in the given file
def read_from_csv(sentence_file):
    data = []

    with open(sentence_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        next(fin)  # skip header row
        for row in reader:
            tmp = SentenceRecord(row[0])
            tmp.processed_sentence = row[1]
            tmp.diag_tag = row[2]
            tmp.sent_tag = row[3]
            tmp.report_id = row[4]
            tmp.report_class = int(row[5])  # report class is dependant on the order in which files are converted to sentences. In future a mapping would be better.
            data.append(tmp)

    return data


# Generate the unlabelled sentence data from the given datafile
def generate_sentences(data_files):
    data = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for i, file_name in enumerate(data_files):
        with open(file_name, 'rb') as file:
            file.readline()  # skip header line
            reader = csv.reader(file)
            for row in reader:
                for sentence in tokenizer.tokenize(row[1]):
                    tmp = SentenceRecord(sentence)
                    tmp.processed_sentence = " ".join(
                        preprocess.textPreprocess(sentence, removeNegationsFromSentences=False))
                    tmp.report_id = row[0]
                    tmp.report_class = i
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


def strip_labels(data):
    for row in data:
        row.diag_tag = ""
        row.sent_tag = ""

    return data


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


def split_data(data, labels, report_ids, split=0.5, shuffle_items=True):
    if not (len(data) == len(labels) == len(report_ids)):
        raise ValueError("data, labels and report ids must be the same size")

    split_count = int(split * len(data))
    data_points = []

    # Create tuples so data can be shuffled
    for i, item in enumerate(data):
        data_points.append((data[i], labels[i], report_ids[i]))

    # Shuffle the data
    if shuffle_items:
        random.shuffle(data_points)

    # Bucket the items into their respective report_ids to avoid overlap
    distinct_report_ids = dict()
    for item in data_points:
        distinct_report_ids.setdefault(item[2], []).append(item)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    count = 0

    # iterate over each bucket
    for key, value in distinct_report_ids.iteritems():
        for item in value:
            if count > split_count:
                train_data.append(item[0])
                train_labels.append(item[1])
            else:
                test_data.append(item[0])
                test_labels.append(item[1])
        # Only update count once the last bucket has finished so that a complete bucket stays in the same output list
        count += len(value)

    return train_data, train_labels, test_data, test_labels


def add_curves(setType, y_true, y_pos_score, subplot_offset, rows=2, cols=2):
    false_positive_rate, true_positive_rate, thresholds_roc = roc_curve(y_true, y_pos_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    pr_avg = average_precision_score(y_true, y_pos_score)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pos_score)

    plt.subplot(rows, cols, subplot_offset)
    plt.title(setType + ' - Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.subplot(rows, cols, subplot_offset + 1)
    plt.title(setType + ' - Precision Recall')
    plt.plot(recall, precision, 'b', label='APS = %0.2f' % pr_avg)
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
