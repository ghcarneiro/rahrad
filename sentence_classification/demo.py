import json
import sys
import data_utils
import numpy as np
import pipelines
from random import shuffle

sys.path.append("..")  # Adds higher directory to python modules path.

import preprocess

usage = "USAGE: " + sys.argv[0] + " type input_file saved_model\n\t type - (input|guess)"
if len(sys.argv) != 4:
    print usage
    sys.exit(1)

game_type = sys.argv[1]
input_file = sys.argv[2]
prebuilt_model_file = sys.argv[3]
model_params = json.load(open(prebuilt_model_file, 'rb'))

if game_type != "input" and game_type != "guess":
    print "Unknown game type. Valid types: input, guess"
    exit(1)

print "Reading data..."
data = data_utils.read_from_csv(input_file)
shuffle(data)
tagged_data = [x for x in data if x.diag_tag == 'n' or x.diag_tag == 'p']
untagged_data = [x for x in data if x.diag_tag == '']

sentences = [x.processed_sentence for x in tagged_data]
labels = [np.float32(x.diag_tag == "p") for x in tagged_data]

print "Building random forest..."
pipe = pipelines.get_count_lsi_randomforest()
pipe.set_params(**model_params)
pipe.fit(sentences, labels)

guess_index = 0
print ""

def input_game():
    print "Write a sentence to see if the system thinks it's diagnostic:"
    sys.stdout.write("> ")
    ans = sys.stdin.readline()
    print ans
    if ans and ans.rstrip() != "":
        ans = ans.rstrip()
        processed_sentence = " ".join(
            preprocess.textPreprocess(ans, removeNegationsFromSentences=False))
        if processed_sentence == "":
            return
        print processed_sentence
        probs = pipe.predict_proba([processed_sentence])[0]
        print probs
        if probs[0] > probs[1]:
            print "[Negative] That output is not diagnostic with a confidence of " + str(probs[0])
        else:
            print "[Positive] That output is diagnostic with a confidence of " + str(probs[1])


def isint(number):
    try:
        val = int(number)
        return True
    except ValueError:
        return False


def guess_game(index):
    lower_s = ""
    lower_p = 0.0
    upper_s = ""
    upper_p = 0.0

    while lower_s == "" or upper_s == "":
        current = untagged_data[index]
        index += 1

        probs = pipe.predict_proba([current.processed_sentence])[0]
        if (probs[0] > 0.70) and probs[0] > lower_p:
            lower_s = current.sentence
            lower_p = probs[0]

        if (probs[1] > 0.70) and probs[1] > upper_p:
            upper_s = current.sentence
            upper_p = probs[1]

    answers = [(lower_s, lower_p, 'n'), (upper_s, upper_p, 'p')]
    shuffle(answers)
    print "Choose which of these two sentences you think is diagnostic:"
    print "\t 1. " + answers[0][0]
    print "\t 2. " + answers[1][0]
    print ""

    sys.stdout.write("> ")
    ans = sys.stdin.readline()
    while not isint(ans) or int(ans) < 1 or int(ans) > 2:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()

    ans = int(ans)
    if answers[ans - 1][2] == 'p':
        print "Correct! The classifier chose the same as you."
    else:
        print "Incorrect. The classifier chose opposite to you. Who's right?"

    return index

if game_type == "guess":
    print str(len(tagged_data)) + " tagged sentences were used to train the classifier."
    print "The classifier has never seen any of the following sentences before."
    print "The machine has learnt what a diagnostic sentence is."
    print ""

while True:
    if game_type == "input":
        input_game()
    elif game_type == "guess":
        guess_index = guess_game(guess_index)
