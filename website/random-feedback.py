# This file randomly assigns a similarity to sentences to mimic gensim without the long wait time.

import sys
import numpy as np
import re
from random import randint

def runReportSimilarity(fileName,threshold=0.7):
    """ Assumes reports have FINDINGS: or REPORT: """
    fileText = [row.rstrip('\n') for row in open(fileName)]
                
    wordsToFind = ["FINDINGS:","REPORT:"]
    report1 = fileText[0]
    report2 = fileText[1]

    startLoc1 = -1
    startLoc2 = -1
    for word in wordsToFind:
        if startLoc1 == -1 and report1.find(word) != -1:
            startLoc1 = report1.find(word)+len(word)
        else:
            startLoc1 = 0
        # In case trainee doesn't write findings/report
        if startLoc2 == -1 and report2.find(word) != -1:
            startLoc2 = report2.find(word)+len(word)
        else:
            startLoc2 = 0

    sCom = []

    report1 = report1[startLoc1:]
    report2 = report2[startLoc2:]
    sentences1 = report1.split('.')
    sentences2 = report2.split('.')
    sent1 = sentences1[:]
    sent2 = sentences2[:]

    correct_sent = 0
    i=0
    output = {'missing': 0, 'corrections': 0, 'extras': 0, 'correct': 0}
    for sentence in sent2:
        r = randint(0,100)
        k = r/100.00
        if k < 0.5:
            k = 0.94
        if k > threshold and correct_sent < len(sent1): 
            s="n\t"+sent2[i]+" "+str(k)+"\t"+str(i)	
            output['correct'] += 1
            correct_sent += 1
            print s
        elif (sentence.strip()) != "": # Checks that string is not empty
            s ="e\t"+sent2[i]+" "+str(k)+"\t"+str(i)
            output['extras'] += 1
            print s
        i+=1
    i=0
    match_sent = 0
    for sentence in sent1:
        r = randint(0,100)
        k = r/100.00
        if k < 0.5:
            k = 0.94
        if k > threshold and match_sent < correct_sent:
            s = "t\t"+sent1[i]+"\t"+str(i)
            match_sent += 1
            print s
	elif len(sent1) - i == correct_sent - match_sent:
            s = "t\t"+sent1[i]+"\t"+str(i)
            match_sent += 1
            print s
        elif (sentence.strip()) != "": # Checks that string is not empty
            output['missing'] += 1
	    s = "m\t"+sent1[i]+"\t"+str(i)
            print s		
        i+=1
			
    return output
			
if __name__ == '__main__':
	# ./similarity input.txt threshold
	# or default threshold
	if (len(sys.argv) < 2):
                print("ERROR: Please specify an input file")
                sys.exit()

	fileName = str(sys.argv[1])
	runReportSimilarity(fileName)
