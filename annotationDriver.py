import preprocess, sys, csv, signal

# Stop the process being killed accidentally so results aren't lost.
def signal_handler(signal, frame):
    print "To exit please press CTL+D so the results can be saved."

signal.signal(signal.SIGINT, signal_handler)

usage = "USAGE: " + sys.argv[0] + " (Brains|CTPA|Plainab|Pvab) (diagnostic|sentiment)"

if len(sys.argv) != 3:
    print usage
    sys.exit(1)

type = sys.argv[1]
tag = sys.argv[2]

dataFile = "./nlp_data/Cleaned" + type + "Full.csv"
sentenceFile = './sentence_label_data/sentences_' + type + '.csv'

# sentences with tags will be represented as lists.
# 0 contains the sentence
# 1 contains the diagnostic tag
# 2 contains the sentiment tag

# Generate the unlabelled sentence data from the given datafile
def generateSentences(dataFile):
    sentenceTags = []
    sentences = preprocess.getSentences([dataFile])

    for sentence in sentences:
        sentenceTags.append([sentence, "", ""])

    return sentenceTags

# Write all sentences to file
def writeToCSV(sentenceFile, sentenceTags):
    with open(sentenceFile, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')

        # Write header 'sentence, labels'
        writer.writerow(["sentence"] + ["diagnostic label"] + ["sentiment label"])
        writer.writerows(sentenceTags)

# Read in all sentences defined in the given file
def readFromCSV(sentenceFile):
    sentenceTags = []

    with open(sentenceFile,'rb') as fin:
        reader = csv.reader(fin, delimiter=",")

        for row in reader:
            sentenceTags.append([row[0], row[1], row[2]])

    # Return the read pairs, but cut off the first row which was headers
    return sentenceTags[1:]

def labelSentences(sentenceTags, tag):
    userExits = False
    for sentence in sentenceTags:
        # If this pair has not been labelled.
        if tag == "diagnostic":
            if sentence[1] is "":
                userExits = getTags(
                    "Is this sentence DIAGNOSTIC, (p)ositive / (n)egative / (u)nsure?",
                    1,
                    ["p", "n", "u"],
                    sentence)
                print ""
        elif tag == "sentiment":
            if sentence[1] is not "" and sentence[2] is "":
                userExits = getTags(
                    "Is the diagnostic OUTCOME (p)ositive / (n)egative / (u)nsure?",
                    2,
                    ["p", "n", "u"],
                    sentence)
                print ""
        else:
            print "Tag not recognised"
            sys.exit(1)

        if userExits:
            break

    if(userExits):
        print "Exiting: Saving progress so far, come back and tag more later."
    else:
        print "Finished tagging! If you were tagging diagnostic, change 'diagnostic' to 'sentiment' to tag the second feature."

    return sentenceTags

def getTags(prompt, index, possibleAnswers, row):
    userExits = False

    print "---> " + row[0]
    print prompt

    while True:
        sys.stdout.write("> ")
        ans = sys.stdin.readline()
        if ans:
            ans = ans.rstrip()
            if ans in possibleAnswers:
                row[index] = ans
                break
            else:
                print "Invalid input. Valid answers: [" + ", ".join(possibleAnswers) + "]"
                continue
        # This block is entered if the user presses CTL+D
        else:
            userExits = True
            break

    return userExits

############# DO NOT UNCOMMENT - WILL OVERWRITE ALL CURRENT DATA #####
# types = ["Brains", "CTPA", "Plainab", "Pvab"]
# for type in types:
#     dataFile = "./nlp_data/Cleaned" + type + "Full.csv"
#     sentenceFile = './sentence_label_data/sentences_' + type + '.csv'
#
#     writeToCSV(sentenceFile, generateSentences(dataFile))
######################################################################

writeToCSV(sentenceFile, labelSentences(readFromCSV(sentenceFile), tag))

