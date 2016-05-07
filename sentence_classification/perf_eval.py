import sys
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import pipelines
import data_utils
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

usage = "USAGE: " + sys.argv[0] + " type split_value data_file [saved_model]"
if len(sys.argv) != 4 and len(sys.argv) != 5:
    print usage
    sys.exit(1)

test_type = sys.argv[1]
split_value = float(sys.argv[2])
data_file = sys.argv[3]
pipe = None
if len(sys.argv) == 5:
    prebuilt_model_file = sys.argv[4]
    pipe = joblib.load(prebuilt_model_file)

data = data_utils.read_from_csv(data_file)

if test_type == "diagnostic":
    filtered_data = [x for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    labels = [np.float32(x.diag_tag == "p") for x in data if x.diag_tag != "" and x.diag_tag != "u"]
elif test_type == "sentiment":
    filtered_data = [x for x in data if x.sent_tag != "" and x.sent_tag != "u"]
    labels = [np.float32(x.sent_tag == "p") for x in data if x.sent_tag != "" and x.sent_tag != "u"]
else:
    raise ValueError("Unknown tag: " + test_type)

data = [x.processed_sentence for x in filtered_data]
report_ids = [x.report_id for x in filtered_data]

train_data, train_labels, test_data, test_labels = data_utils.split_data(data, labels, report_ids, split_value)

# Create transformation pipeline
if pipe is None:
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(train_data, train_labels)

print "Total = " + str(len(filtered_data)) + " [" + str(labels.count(0)) + ", " + str(labels.count(1)) + "]"
print "Train = " + str(len(train_data)) + " [" + str(train_labels.count(0)) + ", " + str(train_labels.count(1)) + "]"
print "Test = " + str(len(test_data)) + " [" + str(test_labels.count(0)) + ", " + str(test_labels.count(1)) + "]"

# Training performance data
y_true_train = train_labels
y_pos_score_train = [pipe.predict_proba([x]).tolist()[0][1] for x in train_data]

# Testing performance data
y_true_test = test_labels
y_pred_test = [pipe.predict([x]).tolist()[0] for x in test_data]
y_pos_score_test = [pipe.predict_proba([x]).tolist()[0][1] for x in test_data]
print classification_report(y_true_test, y_pred_test, target_names=['yes', 'no'])

# Plot both sets of performance curves
plt.figure()
add_curves("Train", y_true_train, y_pos_score_train, 1)
add_curves("Test", y_true_test, y_pos_score_test, 3)
plt.show()

joblib.dump(pipe, 'last_model.pkl', compress=1)
