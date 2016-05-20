import json
import sys, numpy as np
from data_utils import *
import pipelines
from random import shuffle

usage = "USAGE: " + sys.argv[0] + " type passes input_file output_file [saved_model]"
if len(sys.argv) != 5 and len(sys.argv) != 6:
    print usage
    sys.exit(1)

tag_type = sys.argv[1]
passes = int(float(sys.argv[2]))
input_file = sys.argv[3]
output_file = sys.argv[4]
model_params = dict()
if len(sys.argv) == 6:
    prebuilt_model_file = sys.argv[5]
    model_params = json.load(open(prebuilt_model_file, 'rb'))

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

shuffle(data)

# All available positive diagnostics were tagged, so some had to be clobbered
if tag_type == 'sentiment':
    counter = 0
    for item in data:
        if counter > 150:
            item.sent_tag = ""
        if item.sent_tag != "":
            counter += 1

# Each pass contains: classifier training, label prediction, finding the max and placing the respective labels
for i in xrange(passes):
    print ""
    print "PASS " + str(i)

    sentences = [x.processed_sentence for x in data if getattr(x, tag_attr) != "" and getattr(x, tag_attr) != "u"]
    labels = [np.float32(getattr(x, tag_attr) == "p") for x in data if getattr(x, tag_attr) != "" and getattr(x, tag_attr) != "u"]

    print "There are " + str(len(sentences)) + " tagged sentences in this pass"

    # Train model and classifier
    pipe = pipelines.get_count_lsi_randomforest()
    pipe.set_params(**model_params)
    pipe.fit(sentences, labels)


    dflt = SentenceRecord("default")
    dflt.diag_probs = [0.0, 0.0]
    dflt.sent_probs = [0.0, 0.0]

    best_low_p = dflt
    best_low_n = dflt
    best_mid_p = dflt
    best_mid_n = dflt
    best_high_p = dflt
    best_high_n = dflt
    new_pos_this_pass = 0
    new_neg_this_pass = 0

    # Makes predictions for the unlabelled data in the first 1000 items,
    # also keeps track of the max confidence of each classification

    dataset = []
    if tag_type == "diagnostic":
        dataset = [item for item in data[:500] if item.diag_tag == ""]
    else:
        dataset = [item for item in data if item.sent_tag == "" and item.diag_tag == "p"]
    print len(dataset)

    for item in dataset:
        setattr(item, prob_attr, pipe.predict_proba([item.processed_sentence])[0])
        p_prob = getattr(item, prob_attr)[1]
        n_prob = getattr(item, prob_attr)[0]
        if p_prob > 0.9 and p_prob > getattr(best_high_p, prob_attr)[1]:
            best_high_p = item
        elif n_prob > 0.9 and n_prob > getattr(best_high_n, prob_attr)[0]:
            best_high_n = item
        elif 0.8 < p_prob < 0.9 and p_prob > getattr(best_mid_p, prob_attr)[1]:
            best_mid_p = item
        elif 0.8 < n_prob < 0.9 and n_prob > getattr(best_mid_n, prob_attr)[0]:
            best_mid_n = item
        elif 0.7 < p_prob < 0.8 and p_prob > getattr(best_low_p, prob_attr)[1]:
            best_low_p = item
        elif 0.7 < n_prob < 0.8 and n_prob > getattr(best_low_n, prob_attr)[0]:
            best_low_n = item

    if best_high_p is not dflt:
        setattr(best_high_p, tag_attr, 'p')
        new_pos_this_pass += 1
        high_bucket.append((best_high_p, 'p'))
    if best_high_n is not dflt:
        setattr(best_high_n, tag_attr, 'n')
        new_neg_this_pass += 1
        high_bucket.append((best_high_n, 'n'))
    if best_mid_p is not dflt:
        # setattr(best_mid_p, tag_attr, 'p')
        new_pos_this_pass += 1
        mid_bucket.append((best_mid_p, 'p'))
    if best_mid_n is not dflt:
        # setattr(best_mid_n, tag_attr, 'n')
        new_neg_this_pass += 1
        mid_bucket.append((best_mid_n, 'n'))
    if best_low_p is not dflt:
        # setattr(best_low_p, tag_attr, 'p')
        new_pos_this_pass += 1
        low_bucket.append((best_low_p, 'p'))
    if best_low_n is not dflt:
        # setattr(best_low_n, tag_attr, 'n')
        new_neg_this_pass += 1
        low_bucket.append((best_low_n, 'n'))

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
