import sys
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import pipelines
from data_utils import read_from_csv
import numpy as np
from sklearn.externals import joblib


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


usage = "USAGE: " + sys.argv[0] + " type num_train data_file [saved_model]"
if len(sys.argv) != 4 and len(sys.argv) != 5:
    print usage
    sys.exit(1)

test_type = sys.argv[1]
num_train = int(sys.argv[2])
data_file = sys.argv[3]
pipe = None
if len(sys.argv) == 5:
    prebuilt_model_file = sys.argv[4]
    pipe = joblib.load(prebuilt_model_file)

data = read_from_csv(data_file)

if test_type == "diagnostic":
    filtered_data = [x for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    filtered_data = [x for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    labels = [np.float32(x.diag_tag == "p") for x in data if x.diag_tag != "" and x.diag_tag != "u"]
elif test_type == "sentiment":
    filtered_data = [x for x in data if x.sent_tag != "" and x.sent_tag != "u"]
    labels = [np.float32(x.sent_tag == "p") for x in data if x.sent_tag != "" and x.sent_tag != "u"]
else:
    raise ValueError("Unknown tag: " + test_type)

# Extract training data from overall data
train = [x.processed_sentence for x in filtered_data[:num_train]]
train_report_ids = [x.report_id for x in filtered_data[:num_train]]
train_labels = labels[:num_train]

# Create transformation pipeline
if pipe is None:
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(train, train_labels)

test = []
test_labels = []

# Extract test data from overall test data
for i, item in enumerate(filtered_data[num_train:]):
    if item.report_id not in train_report_ids:
        test.append(item.processed_sentence)
        test_labels.append(labels[num_train + i])

print "Total = " + str(len(filtered_data)) + " [" + str(labels.count(0)) + ", " + str(labels.count(1)) + "]"
print "Train = " + str(len(train)) + " [" + str(train_labels.count(0)) + ", " + str(train_labels.count(1)) + "]"
print "Test = " + str(len(test)) + " [" + str(test_labels.count(0)) + ", " + str(test_labels.count(1)) + "]"

# Training performance data
y_true_train = train_labels
y_pos_score_train = [pipe.predict_proba([x]).tolist()[0][1] for x in train]

# Testing performance data
y_true_test = test_labels
y_pred_test = [pipe.predict([x]).tolist()[0] for x in test]
y_pos_score_test = [pipe.predict_proba([x]).tolist()[0][1] for x in test]
print classification_report(y_true_test, y_pred_test, target_names=['yes', 'no'])

# Plot both sets of performance curves
plt.figure()
add_curves("Train", y_true_train, y_pos_score_train, 1)
add_curves("Test", y_true_test, y_pos_score_test, 3)
plt.show()
