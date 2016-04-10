import pipelines
from dataUtils import readFromCSV
import numpy as np

dataFile = './sentence_label_data/sentences_ALL.csv'

data = readFromCSV(dataFile)

taggedSentences = [x.processedSentence for x in data if x.diagTag != "" and x.diagTag != "u"]
labels = [np.float32(x.diagTag == "p") for x in data if x.diagTag != "" and x.diagTag != "u"]

# Extract training data from overall data
train = taggedSentences[:100]
trainLabels = labels[:100]

# Create transformation pipeline
# testPipe = pipelines.get_count_lsi_randomforest()
testPipe = pipelines.get_count_lsi_MLP()
# testPipe = pipelines.get_tfidf_lsi_randomforest()
# testPipe = pipelines.get_count_lsi_SVM()
testPipe.fit(train, trainLabels)

# Extract test data from overall data
test = taggedSentences[101:]
testLabels = labels[101:]

correct = 0

for i, sentence in enumerate(test):
    # print testPipe.predict_proba([sentence])
    # print testLabels[i]
    # print ""
    if testPipe.predict([sentence])[0] == testLabels[i]:
        correct += 1

print correct
print len(test)

print "Accuracy=" + str(float(correct)/len(test))