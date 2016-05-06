import sys, numpy as np
from data_utils import *
import pipelines
from random import shuffle

# Will only tag if the confidence is above these values
POS_THRESHOLD = 0.95
NEG_THRESHOLD = 0.95

usage = "USAGE: " + sys.argv[0] + " type passes input_file output_file"
if len(sys.argv) != 5:
    print usage
    sys.exit(1)

tag_type = sys.argv[1]
passes = int(float(sys.argv[2]))
input_file = sys.argv[3]
output_file = sys.argv[4]

if tag_type != "diagnostic" and tag_type != "sentiment":
    raise ValueError("Unknown tag: " + tag_type)

data = read_from_csv(input_file)


new_pos = 0
new_neg = 0

# Each pass contains: classifier training, label prediction, finding the max and placing the respective labels
for i in xrange(passes):
    print ""
    print "PASS " + str(i)

    sentences = []
    labels = []

    if tag_type == "diagnostic":
        sentences = [x.processed_sentence for x in data if x.diag_tag != "" and x.diag_tag != "u"]
        labels = [np.float32(x.diag_tag == "p") for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    elif tag_type == "sentiment":
        sentences = [x.processed_sentence for x in data if x.sent_tag != "" and x.sent_tag != "u"]
        labels = [np.float32(x.sent_tag == "p") for x in data if x.sent_tag != "" and x.sent_tag != "u"]

    print "There are " + str(len(sentences)) + " tagged sentences in this pass"

    # Train model and classifier
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(sentences, labels)

    best_pos = 0
    best_neg = 0
    new_pos_this_pass = 0
    new_neg_this_pass = 0

    # Makes predicitions for the unlabelled data in the first 1000 items,
    # also keeps track of the max confidence of each classification
    if tag_type == "diagnostic":
        for item in [item for item in data[:500] if item.diag_tag == ""]:
            item.diag_probs = pipe.predict_proba([item.processed_sentence])[0]
            best_neg = max(item.diag_probs[0], best_neg)
            best_pos = max(item.diag_probs[1], best_pos)
    elif tag_type == "sentiment":
        for item in [item for item in data[:500] if item.sent_tag == ""]:
            item.sentProbs = pipe.predict_proba([item.processed_sentence])[0]
            best_neg = max(item.sentProbs[0], best_neg)
            best_pos = max(item.sentProbs[1], best_pos)

    print "Best positive confidence: " + str(best_pos)
    print "Best negative confidence: " + str(best_neg)

    # Allocates the labels for each of the determined max values if they lie above the threshold
    if tag_type == "diagnostic":
        for item in [item for item in data[:500] if item.diag_tag == ""]:
            if POS_THRESHOLD < best_pos == item.diag_probs[1]:
                item.diag_tag = "p"
                new_pos_this_pass += 1
            if NEG_THRESHOLD < best_neg == item.diag_probs[0]:
                item.diag_tag = "n"
                new_neg_this_pass += 1
    elif tag_type == "sentiment":
        for item in [item for item in data[:500] if item.sent_tag == ""]:
            if POS_THRESHOLD < best_pos == item.sentProbs[1]:
                item.sent_tag = "p"
                new_pos_this_pass += 1
            if NEG_THRESHOLD < best_neg == item.sentProbs[0]:
                item.sent_tag = "n"
                new_neg_this_pass += 1

    # Shuffle the data to ensure a new set of 1000 is served up on the next pass
    shuffle(data)

    new_pos += new_pos_this_pass
    new_neg += new_neg_this_pass

    print str(new_pos_this_pass) + " new positive tags added this pass"
    print str(new_neg_this_pass) + " new negative tags added this pass"

# Save the ouput of the automatic learning
write_to_csv(output_file, data)

print ""
print "Completed " + str(passes) + " passes over the data."
print str(new_pos) + " new positive tags added"
print str(new_neg) + " new negative tags added"
