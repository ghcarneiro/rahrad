import sys
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
from sklearn.ensemble import RandomForestClassifier


class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.processed_sentence = ""
        self.diag_probs = []
        self.sent_probs = []
        self.diag_tag = ""
        self.sent_tag = ""
        self.report_id = ""
        self.feature_vector = []


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


usage = "USAGE: " + sys.argv[0] + " split_value data_file"
if len(sys.argv) != 3:
    print usage
    sys.exit(1)

split_value = float(sys.argv[1])
data_file = sys.argv[2]

# load data file
data = pkl.load(open(data_file, "rb"))

# extract features and labels
features = [x.feature_vector for x in data]
labels = [np.float32(x.sent_tag == "p") for x in data]

# Calculate split and split data
split_thresh = int(len(features) * split_value)

train_data = features[:split_thresh]
train_labels = labels[:split_thresh]
test_data = features[split_thresh:]
test_labels = labels[split_thresh:]

# Set random forest parameters
random_forest_params = {"min_samples_leaf": 80, "n_estimators": 1000, "max_depth": 2, "min_samples_split": 30}
#random_forest_params = {"min_samples_leaf": 1, "n_estimators": 1000, "max_depth": 100000, "min_samples_split": 1}

# Create random forest and train
rf = RandomForestClassifier(**random_forest_params)
rf.fit(train_data, train_labels)

print "Total = " + str(len(features)) + " [" + str(labels.count(0)) + ", " + str(labels.count(1)) + "]"
print "Train = " + str(len(train_data)) + " [" + str(train_labels.count(0)) + ", " + str(train_labels.count(1)) + "]"
print "Test = " + str(len(test_data)) + " [" + str(test_labels.count(0)) + ", " + str(test_labels.count(1)) + "]"

# Training performance data
y_true_train = train_labels
y_pos_score_train = [rf.predict_proba([x]).tolist()[0][1] for x in train_data]

# Testing performance data
y_true_test = test_labels
y_pos_score_test = [rf.predict_proba([x]).tolist()[0][1] for x in test_data]

# Plot both sets of performance curves
plt.figure()
add_curves("Train", y_true_train, y_pos_score_train, 1)
add_curves("Test", y_true_test, y_pos_score_test, 3)

plt.show()
