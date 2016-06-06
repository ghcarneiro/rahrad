import signal
from random import shuffle

import numpy as np

import pipelines
from data_utils import *


# Stop the process being killed accidentally so results aren't lost.
def signal_handler(signal, frame):
    print "To exit please press CTL+D so the results can be saved."


def label_sentences(list_sentence_record, tag_to_label, ignore_probs):
    """
    Control loop for labelling sentences. Takes in a list of SentenceRecords and labels each until probability
     difference threshold is reached, no more data is available or user quits.
    :param list_sentence_record: List of SentenceRecord objects to collect tags for
    :param tag_to_label: Which tag type to label (diagnostic/sentiment)
    :param ignore_probs: If true ignores the probability difference threshold as an exit condition.
    :return: List of SentenceRecord objects with labels applied.
    """
    user_exits = False
    probability_difference = 0

    # Ensures that the datatype in the list is SentenceRecord
    if not isinstance(list_sentence_record[0], SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    for sentence in list_sentence_record:
        if tag_to_label == "diagnostic" and sentence.diag_tag == "" and len(sentence.diag_probs) != 0 or ignore_probs:
            if not ignore_probs:
                probability_difference = abs(sentence.diag_probs[0] - sentence.diag_probs[1])
            user_exits, ans = get_tags(
                "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                ["p", "n", "u"],
                sentence)
            print ""

            sentence.diag_tag = ans

        elif tag_to_label == "sentiment" and sentence.diag_tag == "p" and sentence.sent_tag == "" and len(
                sentence.sent_probs) != 0 or ignore_probs:
            if not ignore_probs:
                probability_difference = abs(sentence.sent_probs[0] - sentence.sent_probs[1])
            user_exits, ans = get_tags(
                "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                ["p", "n", "u"],
                sentence)
            print ""

            sentence.sent_tag = ans

        if not ignore_probs and probability_difference > diff_tolerance:
            print "Uncertainty tolerance reached. Rerun to classify more."
            user_exits = True

        if user_exits:
            break

    if (user_exits):
        print "Exiting: Saving progress so far, come back and tag more later."
    else:
        print "Finished tagging! If you were tagging diagnostic, " \
              "change 'diagnostic' to 'sentiment' to tag the second feature."

    return list_sentence_record


# Controls tagging of single sentence
# Returns whether the user wants to exit and the tag chosen
def get_tags(prompt, possible_answers, current_sentence_record):
    """
    Controls the tagging of a single sentence, issuing a prompt and looping until exit or acceptable input is received.
    :param prompt: The question presented to the user.
    :param possible_answers: List of acceptable answers from the user.
    :param current_sentence_record: Current SentenceRecord that is being tagged.
    :return: Tuple containing whether the user chose to exit the application and the tag that was inputted.
    """
    user_exits = False

    if not isinstance(current_sentence_record, SentenceRecord):
        raise ValueError("Expected SentenceRecord")

    print "---> " + current_sentence_record.sentence
    print prompt

    while True:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()
        if ans:
            ans = ans.rstrip()
            if ans == "quit":
                user_exits = True
                break
            else:
                if ans in possible_answers:
                    return user_exits, ans
                else:
                    print "Invalid input. Valid answers: [" + ", ".join(possible_answers) + "]"
                    continue
        # This block is entered if the user presses CTL+D
        else:
            user_exits = True
            break

    return user_exits, ""


# Custom comparison function for diagnostic label probability for two SentenceRecord objects.
def compare_diag_prob(item1, item2):
    diff1 = 1 if item1.diag_probs == [] else abs(item1.diag_probs[0] - item1.diag_probs[1])
    diff2 = 1 if item2.diag_probs == [] else abs(item2.diag_probs[0] - item2.diag_probs[1])

    if diff1 == diff2:
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1


# Custom comparison function for sentiment label probability for two SentenceRecord objects.
def compare_sent_prob(item1, item2):
    diff1 = 1 if item1.sent_probs == [] else abs(item1.sent_probs[0] - item1.sent_probs[1])
    diff2 = 1 if item2.sent_probs == [] else abs(item2.sent_probs[0] - item2.sent_probs[1])

    if diff1 == diff2:
        return 0
    elif diff1 < diff2:
        return -1
    else:
        return 1


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) != 3:
        print "USAGE: " + sys.argv[0] + " (diagnostic|sentiment) data_file"
        sys.exit(1)

    tag = sys.argv[1]
    data_file = sys.argv[2]

    if tag == "diagnostic":
        diff_tolerance = 0.15
    elif tag == "sentiment":
        diff_tolerance = 0.5

    # Generates the list of SentenceRecords from original data.
    # WILL DESTROY EXISTING TAGS
    if tag == "generate":
        generate_sentences_from_raw()
        exit(0)

    ########################
    ### Data Retrieval   ###
    ########################

    print "Reading data file"
    data = read_from_csv(data_file)

    ########################
    ### Classification   ###
    ########################

    tagged_sentences = []
    labels = []

    ignore_probs = False

    # Extract just the sentences that are tagged and were not unsure and preprocess them
    print "Extracting tagged sentences for classification"
    if tag == "diagnostic":
        tagged_sentences = [x.processed_sentence for x in data if x.diag_tag != "" and x.diag_tag != "u"]
        labels = [np.float32(x.diag_tag == "p") for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    elif tag == "sentiment":
        tagged_sentences = [x.processed_sentence for x in data if x.sent_tag != "" and x.sent_tag != "u"]
        labels = [np.float32(x.sent_tag == "p") for x in data if x.sent_tag != "" and x.sent_tag != "u"]
    else:
        raise ValueError("Unknown tag: " + tag)

    if len(tagged_sentences) != 0:
        print "Number of tagged sentences: " + str(len(tagged_sentences))
        print "Building feature extraction model and classifier"
        pipe = pipelines.get_count_lsi_randomforest()
        pipe.fit(tagged_sentences, labels)

        # Take smaller working set, no need to classify everything
        working_list = [x for x in data if x not in tagged_sentences]
        if len(working_list) > 300:
            working_list = working_list[0:299]

        print "Calculating classifications"
        if tag == "diagnostic":
            for row in working_list:
                row.diag_probs = pipe.predict_proba([row.processed_sentence])[0]
        elif tag == "sentiment":
            for row in working_list:
                row.sent_probs = pipe.predict_proba([row.processed_sentence])[0]

        print "Sorting data"
        if tag == "diagnostic":
            data.sort(cmp=compare_diag_prob)
        elif tag == "sentiment":
            data.sort(cmp=compare_sent_prob)
    else:
        ignore_probs = True
        print "Classification skipped, there are no tagged sentences"

    #################
    ### Labelling ###
    #################
    data = label_sentences(data, tag, ignore_probs)

    print "Saving data"
    shuffle(data)
    write_to_csv(data_file, data)
