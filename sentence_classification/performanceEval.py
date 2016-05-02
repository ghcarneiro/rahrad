import sys
from sklearn.metrics import classification_report, roc_curve, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

import pipelines
from dataUtils import readFromCSV
import numpy as np

usage = "USAGE: " + sys.argv[0] + "type numTrain dataFile"
if len(sys.argv) != 4:
    print usage
    sys.exit(1)

type = sys.argv[1]
numTrain = int(sys.argv[2])
dataFile = sys.argv[3]

data = readFromCSV(dataFile)

if type == "diagnostic":
    filteredData = [x for x in data if x.diagTag != "" and x.diagTag != "u"]
    labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != "" and x.diagTag != "u"]
elif type == "sentiment":
    filteredData = [x for x in data if x.sentTag != "" and x.sentTag != "u"]
    labels = [np.float32(x.sentTag == "p") for x in data if x.sentTag != "" and x.sentTag != "u"]
else:
    raise ValueError("Unknown tag: " + type)

# Extract training data from overall data
train = [x.processedSentence for x in filteredData[:numTrain]]
trainReportIDs = [x.reportID for x in filteredData[:numTrain]]
trainLabels = labels[:numTrain]

# Create transformation pipeline
# testPipe = pipelines.get_count_randomforest()
testPipe = pipelines.get_count_lsi_randomforest()
# testPipe = pipelines.get_tfidf_lsi_randomforest()
# testPipe = pipelines.get_count_lsi_SVM()
testPipe.fit(train, trainLabels)

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

y_true = testLabels
y_pred = [testPipe.predict([x]).tolist()[0] for x in test]
y_pos_score = [testPipe.predict_proba([x]).tolist()[0][1] for x in test]

print y_true
print y_pred

print classification_report(y_true, y_pred, target_names=['yes', 'no'])

false_positive_rate, true_positive_rate, thresholds_roc = roc_curve(y_true, y_pos_score)
roc_auc = auc(false_positive_rate, true_positive_rate)

pr_avg = average_precision_score(y_true, y_pos_score)

precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pos_score)

plt.figure()

# Plot ROC curve
plt.subplot(121)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.subplot(122)
plt.title('Precision Recall')
plt.plot(recall, precision, 'b', label='APS = %0.2f' % pr_avg)
plt.legend(loc='lower right')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('Precision')
plt.xlabel('Recall')

plt.show()
