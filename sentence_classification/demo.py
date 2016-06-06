import json
import sys
from random import shuffle
import numpy as np
import data_utils
import pipelines

sys.path.append("..")  # Adds higher directory to python modules path.

import preprocess


def input_game():
    """
    This game accepts a 'report sentence' from a user and attempts to classify it.
    :return: None
    """
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


def guess_game(index_so_far):
    """
    This game presents the user with two sentences and asks them to classify the diagnostic sentence.
    Says if the user agreed or disagreed with the classifier.
    :param index_so_far:
    :return:
    """
    lower_sentence = ""
    lower_probability = 0.0
    upper_sentence = ""
    upper_probability = 0.0

    while lower_sentence == "" or upper_sentence == "":
        current = untagged_data[index_so_far]
        index_so_far += 1

        probs = pipe.predict_proba([current.processed_sentence])[0]
        if (probs[0] > 0.70) and probs[0] > lower_probability:
            lower_sentence = current.sentence
            lower_probability = probs[0]

        if (probs[1] > 0.70) and probs[1] > upper_probability:
            upper_sentence = current.sentence
            upper_probability = probs[1]

    answers = [(lower_sentence, lower_probability, 'n'), (upper_sentence, upper_probability, 'p')]
    shuffle(answers)
    print "Choose which of these two sentences you think is diagnostic:"
    print "\t 1. " + answers[0][0]
    print "\t 2. " + answers[1][0]
    print ""

    sys.stdout.write("> ")
    ans = sys.stdin.readline()
    while not data_utils.isint(ans) or int(ans) < 1 or int(ans) > 2:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()

    ans = int(ans)
    if answers[ans - 1][2] == 'p':
        print "Correct! The classifier chose the same as you."
    else:
        print "Incorrect. The classifier chose opposite to you. Who's right?"

    return index_so_far


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "USAGE: " + sys.argv[0] + " type input_file saved_model\n\t type - (input|guess)"
        sys.exit(1)

    # Read in command line parameters
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

    if game_type == "guess":
        print str(len(tagged_data)) + " tagged sentences were used to train the classifier."
        print "The classifier has never seen any of the following sentences before."
        print "The machine has learnt what a diagnostic sentence is."
        print ""

    # Game loop, runs until killed.
    while True:
        if game_type == "input":
            input_game()
        elif game_type == "guess":
            guess_index = guess_game(guess_index)
