import search_engine2 as search_engine
import gensim
import sys
import numpy as np

def runReportSimilarity(fileName,tfidf_model,lsi_model,dictionary,threshold=0.9,reportType="lsi"):
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

    if reportType == "lsi":
        report1 = search_engine.textPreprocess(report1)
        report1 = search_engine.getDerivations(report1, dictionary)
        report2 = search_engine.textPreprocess(report2)
        report2 = search_engine.getDerivations(report2, dictionary)
        for i in range(len(sentences1)):
            sentences1[i] = search_engine.textPreprocess(sentences1[i])
            sentences1[i] = search_engine.getDerivations(sentences1[i], dictionary)
        for i in range(len(sentences2)):
            sentences2[i] = search_engine.textPreprocess(sentences2[i])
            sentences2[i] = search_engine.getDerivations(sentences2[i], dictionary)

		#corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
        vec_lsi1 = lsi_model[tfidf_model[dictionary.doc2bow(report1)]]
        vec_lsi2 = lsi_model[tfidf_model[dictionary.doc2bow(report2)]]
        sen1Corp = [dictionary.doc2bow(sent) for sent in sentences1]
        sen2Corp = [dictionary.doc2bow(sent) for sent in sentences2]
        vec_lsis1 = lsi_model[tfidf_model[sen1Corp]]
        vec_lsis2 = lsi_model[tfidf_model[sen2Corp]]

		# print corpus.num_terms
		# ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=corpus.num_terms)
        ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=10)
		# similarity table
        for  i in vec_lsis2:
            sCom.append(ind[i])
    elif reportType == "rnn":
        pythonpleaser = "3"

    missing = [0 for s in sent1]
	# obtain correct sentence
    i = 0


    output = {'missing': 0, 'corrections': 0, 'extras': 0, 'correct': 0}
    for col in sCom:
	#for col in range(len(sCom[0]))
	#for col in sent2: 
        threshold_value = 0
        aboveTopThreshold = False 
        j = 0
        bestSim = 0
        for sim in col:
            if sim > threshold: 
                aboveTopThreshold = True 
            if sim > bestSim:
                bestSim = sim
            if missing[j] < sim:
                missing[j] = sim 
            j+=1

        threshold_value = bestSim
   
        if aboveTopThreshold: 
			#maybe add percentage for debugging
			#sent2[i] = " ".join([k for k in sent2[i]])
            s="n\t"+sent2[i]+" "+str(threshold_value)+"\t"	
            output['correct'] += 1
            print s
        elif (threshold_value > 0): # Exclude blank strings
			#sent2[i] = " ".join([k for k in sent2[i]])
            s ="e\t"+sent2[i]+" "+str(threshold_value)+"\t"
            output['extras'] += 1
            print s
        i+=1
    i=0
    for k in missing:
        if k <= threshold:
			#sent1[i] = " ".join([k for k in sent1[i]])
			#s = str(k)
            s = "m\t"+sent1[i]+"\t"
            output['missing'] += 1
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
