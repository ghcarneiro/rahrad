import sys, numpy as np
from data_utils import *
import pipelines
from random import shuffle

# Will only tag if the confidence is above these values
# POS_THRESHOLD = 0.95
# NEG_THRESHOLD = 0.95

usage = "USAGE: " + sys.argv[0] + " type passes input_file output_file"
if len(sys.argv) != 5:
    print usage
    sys.exit(1)

tag_type = sys.argv[1]
passes = int(float(sys.argv[2]))
input_file = sys.argv[3]
output_file = sys.argv[4]

if tag_type == 'diagnostic':
    tag_attr = 'diag_tag'
    prob_attr = 'diag_probs'
elif tag_type == 'sentiment':
    tag_attr = 'sent_tag'
    prob_attr = 'sent_probs'
else:
    raise ValueError("Unknown tag: " + tag_type)

data = read_from_csv(input_file)

new_pos = 0
new_neg = 0

# BUCKETS
low_bucket = []  # 70-80
mid_bucket = []  # 80-90
high_bucket = []  # 90-100


# Each pass contains: classifier training, label prediction, finding the max and placing the respective labels
for i in xrange(passes):
    print ""
    print "PASS " + str(i)

    sentences = [x.processed_sentence for x in data if getattr(x, tag_attr) != "" and  getattr(x, tag_attr) != "u"]
    labels = [np.float32(x.diag_tag == "p") for x in data if getattr(x, tag_attr) != "" and getattr(x, tag_attr) != "u"]

    print "There are " + str(len(sentences)) + " tagged sentences in this pass"

    # Train model and classifier
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.fit(sentences, labels)

    dflt = SentenceRecord("default")
    dflt.diag_probs = [0.0, 0.0]
    dflt.sent_probs = [0.0, 0.0]

    best_low = dflt
    best_mid = dflt
    best_high = dflt
    new_pos_this_pass = 0
    new_neg_this_pass = 0

    # Makes predicitions for the unlabelled data in the first 1000 items,
    # also keeps track of the max confidence of each classification

    for item in [item for item in data[:500] if item.diag_tag == ""]:
        setattr(item, prob_attr, pipe.predict_proba([item.processed_sentence])[0])
        p_prob = getattr(item, prob_attr)[0]
        n_prob = getattr(item, prob_attr)[1]
        if p_prob > 0.9 or n_prob > 0.9:
            if max(p_prob, n_prob) > max(getattr(best_high, prob_attr)[0], getattr(best_high, prob_attr)[1]):
                best_high = item
        elif p_prob > 0.8 or n_prob > 0.8:
            if max(p_prob, n_prob) > max(getattr(best_mid, prob_attr)[0], getattr(best_mid, prob_attr)[1]):
                best_mid = item
        elif p_prob > 0.7 or n_prob > 0.7:
            if max(p_prob, n_prob) > max(getattr(best_low, prob_attr)[0], getattr(best_low, prob_attr)[1]):
                best_low = item

    if best_high is not dflt:
        if getattr(best_high, prob_attr)[0] > getattr(best_high, prob_attr)[1]:
            setattr(best_high, tag_attr, 'n')
            new_neg_this_pass += 1
            high_bucket.append((best_high, 'n'))
        else:
            setattr(best_high, tag_attr, 'p')
            new_pos_this_pass += 1
            high_bucket.append((best_high, 'p'))
    if best_mid is not dflt:
        if getattr(best_mid, prob_attr)[0] > getattr(best_mid, prob_attr)[1]:
            new_neg_this_pass += 1
            mid_bucket.append((best_mid, 'n'))
        else:
            new_pos_this_pass += 1
            mid_bucket.append((best_mid, 'p'))
    if best_low is not dflt:
        if getattr(best_low, prob_attr)[0] > getattr(best_low, prob_attr)[1]:
            new_neg_this_pass += 1
            low_bucket.append((best_low, 'n'))
        else:
            new_pos_this_pass += 1
            low_bucket.append((best_low, 'p'))

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

print ""
print tag_type
print ""

print "High bucket: (90-100)"
for item in high_bucket:
    print str(item[0].sentence) + ", " + str(getattr(item[0], prob_attr)) + ", " + str(item[1])
print ""

print "Mid bucket: (80-90)"
for item in mid_bucket:
    print str(item[0].sentence) + ", " + str(getattr(item[0], prob_attr)) + ", " + str(item[1])
print ""

print "Low bucket: (70-80)"
for item in low_bucket:
    print str(item[0].sentence) + ", " + str(getattr(item[0], prob_attr)) + ", " + str(item[1])
print ""
