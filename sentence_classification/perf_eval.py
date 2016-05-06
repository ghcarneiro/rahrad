import sys
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import pipelines
from data_utils import readFromCSV
import numpy as np
from sklearn.externals import joblib

def addCurves(setType, y_true, y_pos_score, subplotOffset, rows=2, cols=2):
    false_positive_rate, true_positive_rate, thresholds_roc = roc_curve(y_true, y_pos_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    pr_avg = average_precision_score(y_true, y_pos_score)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pos_score)

    plt.subplot(rows, cols, subplotOffset)
    plt.title(setType + ' - Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.subplot(rows, cols, subplotOffset + 1)
    plt.title(setType + ' - Precision Recall')
    plt.plot(recall, precision, 'b', label='APS = %0.2f' % pr_avg)
    plt.legend(loc='lower right')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Precision')
    plt.xlabel('Recall')

usage = "USAGE: " + sys.argv[0] + " type numTrain dataFile [savedModel]"
if len(sys.argv) != 4 and len(sys.argv) != 5:
    print usage
    sys.exit(1)

testType = sys.argv[1]
numTrain = int(sys.argv[2])
dataFile = sys.argv[3]
pipe = None
if len(sys.argv) == 5:
    prebuiltModelFile = sys.argv[4]
    pipe = joblib.load(prebuiltModelFile)

data = readFromCSV(dataFile)

if testType == "diagnostic":
    filteredData = [x for x in data if x.diagTag != "" and x.diagTag != "u"]
    labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != "" and x.diagTag != "u"]
elif testType == "sentiment":
    filteredData = [x for x in data if x.sentTag != "" and x.sentTag != "u"]
    labels = [np.float32(x.sentTag == "p") for x in data if x.sentTag != "" and x.sentTag != "u"]
else:
    raise ValueError("Unknown tag: " + testType)

# Extract training data from overall data
train = [x.processedSentence for x in filteredData[:numTrain]]
trainReportIDs = [x.reportID for x in filteredData[:numTrain]]
trainLabels = labels[:numTrain]

# Create transformation pipeline
if pipe is None:
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(train, trainLabels)

test = []
testLabels = []

# Extract test data from overall test data
for i, item in enumerate(filteredData[numTrain:]):
    if item.reportID not in trainReportIDs:
        test.append(item.processedSentence)
        testLabels.append(labels[numTrain + i])

print "Total = " + str(len(filteredData)) + " [" + str(labels.count(0)) + ", " + str(labels.count(1)) + "]"
print "Train = " + str(len(train)) + " [" + str(trainLabels.count(0)) + ", " + str(trainLabels.count(1)) + "]"
print "Test = " + str(len(test)) + " [" + str(testLabels.count(0)) + ", " + str(testLabels.count(1)) + "]"

# Training performance data
y_true_train = trainLabels
y_pos_score_train = [pipe.predict_proba([x]).tolist()[0][1] for x in train]

# Testing performance data
y_true_test = testLabels
y_pred_test = [pipe.predict([x]).tolist()[0] for x in test]
y_pos_score_test = [pipe.predict_proba([x]).tolist()[0][1] for x in test]
print classification_report(y_true_test, y_pred_test, target_names=['yes', 'no'])

# Plot both sets of performance curves
plt.figure()
addCurves("Train", y_true_train, y_pos_score_train, 1)
addCurves("Test", y_true_test, y_pos_score_test, 3)
plt.show()
