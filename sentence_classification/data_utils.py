import csv
import random
import sys
import matplotlib.pyplot as plt
import nltk
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

sys.path.append("..")  # Adds higher directory to python modules path.

import preprocess


# Data container for all operations in the project.
# Is serialised with csv for speed.
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


def write_to_csv(output_sentence_file, list_sentence_record):
    """
    Writes the given list of SentenceRecord objects to csv file with the given name
    :param output_sentence_file: File name to write to
    :param list_sentence_record: List of SentenceRecord objects
    """
    with open(output_sentence_file, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header row
        writer.writerow(
            ["sentence"] + ["processed sentence"] + ["diagnostic label"] + ["sentiment label"] + ["report id"] + [
                "report class"])
        for row in list_sentence_record:
            writer.writerow(
                [row.sentence] + [row.processed_sentence] + [row.diag_tag] + [row.sent_tag] + [row.report_id] + [
                    row.report_class])


def read_from_csv(input_sentence_file):
    """
    Read a list of SentenceRecord objects from the given file in the same csv structure as in write_to_csv
    Assumes the first row is a header and skips.
    :param input_sentence_file: The file to read from
    :return: List of SentenceRecord objects
    """
    data = []

    with open(input_sentence_file, 'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        next(fin)  # skip header row
        for row in reader:
            tmp = SentenceRecord(row[0])
            tmp.processed_sentence = row[1]
            tmp.diag_tag = row[2]
            tmp.sent_tag = row[3]
            tmp.report_id = row[4]
            tmp.report_class = int(row[
                                       5])  # report class is dependant on the order in which files are converted to sentences. In future a mapping would be better.
            data.append(tmp)

    return data


# Generate the unlabelled sentence data from the given datafile
def generate_sentences(data_files):
    """
    Generates SentenceRecord objects using the given datafiles. Tokenises reports into sentences and then preprocesses
    each sentence. Also saves the report_id and report class into the object.
    Note: Currently report class is just the index of the datafile extracted from, a global mapping would be better.
    :param data_files: List of strings corresponding the raw csv datafiles in the format 'report_id, report'
    :return: List of SentenceRecord objects extracted from the files.
    """
    list_sentence_record = []
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
                    list_sentence_record.append(tmp)

    return list_sentence_record


def remove_duplicates(list_sentence_record):
    """
    Removes duplicate SentenceRecord objects. Uses the processed_sentence as the check for uniqueness.
    :param list_sentence_record: List of SentenceRecords to remove duplicates from.
    :return: Original list with duplicates removed.
    """
    added = set()
    res = []

    for row in list_sentence_record:
        if row.processedSentence not in added:
            res.append(row)
            added.add(row.processedSentence)

    return res


def strip_labels(list_sentence_record):
    """
    Sets the diagnostic and sentiment tags from the given list of SentenceRecord objects to empty string ""
    :param list_sentence_record: List of SentenceRecords to remove labels from.
    :return: Original list with labels removed.
    """
    for row in list_sentence_record:
        row.diag_tag = ""
        row.sent_tag = ""

    return list_sentence_record


def generate_sentences_from_raw():
    """
    Generates a list of SentenceRecord objects from the default set of report files and writes them to file
    (./sentence_label_data/sentences_ALL.csv)
    """
    confirm = raw_input("Are you sure you want to regenerate? (yes/no) ")
    if confirm == "yes":

        files = ["./sentence_label_data/CleanedBrainsFull.csv",
                 "./sentence_label_data/CleanedCTPAFull.csv",
                 "./sentence_label_data/CleanedPlainabFull.csv",
                 "./sentence_label_data/CleanedPvabFull.csv"]

        output_file = './sentence_label_data/sentences_generated' \
                      '.csv'
        write_to_csv(output_file, remove_duplicates(generate_sentences(files)))
    else:
        print "Cancelled."


def split_data(features, labels, report_ids, split=0.5, shuffle_items=True):
    """
    Splits the given data into training and testing sets, with the split and shuffle of items specified.
    This function also takes in a list of report_ids so that the two sets dont overlaps in report_ids they contain.
    Features, labels and report_ids are separate lists where the entry with the same index in each is connected,
    therefore must have the same length.
    :param features: List of features (Features can be any data type as long as they are all in a list)
    :param labels: List of labels corresponding to features.
    :param report_ids: Corresponding report_ids for each feature
    :param split: The percentage of the set to be training, the spit proportion.
    :param shuffle_items: Whether to shuffle the set of items before splitting.
    :return: Tuple of lists corresponding to train and test, features and labels (train_data, train_labels, test_data, test_labels)
    """

    # Ensure all lists are of equal size
    if not (len(features) == len(labels) == len(report_ids)):
        raise ValueError("features, labels and report ids must be the same size")

    split_count = int(split * len(features))
    data_points = []

    # Create tuples so data can be shuffled
    for i, item in enumerate(features):
        data_points.append((features[i], labels[i], report_ids[i]))

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


def add_curves(set_type, y_true, y_pos_score, subplot_offset, rows=2, cols=2):
    """
    Adds an ROC and PR curve to a set of graphs at the given offset.
    :param set_type: Specifies the title for this set of data, train or test
    :param y_true: List of correct labels
    :param y_pos_score: List of probabilities of the positive classification
    :param subplot_offset: Offset for which subplot this starts at (for doing multiple subplot graphs)
    :param rows: Number of rows the overall plot is (default 2)
    :param cols: Number of columns the overall plot is (default 2)
    """
    false_positive_rate, true_positive_rate, thresholds_roc = roc_curve(y_true, y_pos_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    pr_avg = average_precision_score(y_true, y_pos_score)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pos_score)

    plt.subplot(rows, cols, subplot_offset)
    plt.title(set_type + ' - Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.subplot(rows, cols, subplot_offset + 1)
    plt.title(set_type + ' - Precision Recall')
    plt.plot(recall, precision, 'b', label='APS = %0.2f' % pr_avg)
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Precision')
    plt.xlabel('Recall')


def isint(number):
    """
    Helper function to determine if a number is an integer
    :param number: Number to test
    :return: True if number is an integer
    """
    try:
        val = int(number)
        return True
    except ValueError:
        return False
