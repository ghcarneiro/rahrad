import search_engine
import time
import random
import gensim
import itertools
import sys
def runReportSimilarity2(fileName,threshold=0.9):
        """ Assumes reports have FINDINGS: or REPORT: """
        fileText = [row.rstrip('\n') for row in open(fileName)]
                
        wordsToFind = ["FINDINGS:","REPORT:"]
        report1 = fileText[0]
        report2 = fileText[1]

        startLoc1 = -1
        #startLoc2 = -1
        for word in wordsToFind:
                if startLoc1 == -1 and report1.find(word) != -1:
                        startLoc1 = report1.find(word)+len(word)
                #if startLoc2 == -1 and report2.find(word) != -1:
                #       startLoc2 = report2.find(word)+len(word)

        report1 = report1[startLoc1:]
        #report2 = report2[startLoc2:]
        sentences1 = report1.split('.')
        sent1 = sentences1[:]
        sentences2 = report2.split('.')
        sent2 = sentences2[:]
                
        report1 = search_engine.textPreprocess(report1)
        report1 = search_engine.getDerivations(report1)
        report2 = search_engine.textPreprocess(report2)
        report2 = search_engine.getDerivations(report2)
        for i in range(len(sentences1)):
                sentences1[i] = search_engine.textPreprocess(sentences1[i])
                sentences1[i] = search_engine.getDerivations(sentences1[i])
        for i in range(len(sentences2)):
                sentences2[i] = search_engine.textPreprocess(sentences2[i])
                sentences2[i] = search_engine.getDerivations(sentences2[i])

        #corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
        tfidf_model = gensim.models.TfidfModel.load('./model_files/reports.tfidf_model')
        lsi_model = gensim.models.LsiModel.load('./model_files/reports.lsi_model')

        dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
        vec_lsi1 = lsi_model[tfidf_model[dictionary.doc2bow(report1)]]
        vec_lsi2 = lsi_model[tfidf_model[dictionary.doc2bow(report2)]]
        sen1Corp = [dictionary.doc2bow(sent) for sent in sentences1]
        sen2Corp = [dictionary.doc2bow(sent) for sent in sentences2]
        vec_lsis1 = lsi_model[tfidf_model[sen1Corp]]
        vec_lsis2 = lsi_model[tfidf_model[sen2Corp]]
        sCom = []

	# print corpus.num_terms
        # ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=corpus.num_terms)
        ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=10)
	# similarity table
        for  i in vec_lsis2:
                sCom.append(ind[i])
	missing = [0 for s in vec_lsis1]
	# obtain correct sentence
	i = 0

	# correction is a wrong sentence but has close meaning 
	# to another sentence therefore we can give a suggestion for a correction

	output = {'missing': 0, 'corrections': 0, 'wrong': 0, 'correct': 0}
	for col in sCom:
		aboveTopThreshold = False 
		#aboveMedThreshold = False
		j = 0
		correction = ""
		bestSim = 0
		for sim in col:
			if sim > threshold: 
				aboveTopThreshold = True 
			elif sim > threshold*0.9:
				correction = sent1[j]
				aboveMedThreshold = True

			if sim > bestSim:
				bestSim = sim
			if missing[j] < sim:
				missing[j] = sim
				
			j+=1
		if aboveTopThreshold: 
			#s = str(bestSim)
			s="n\t"+sent2[i]+"\t"	
			output['correct'] += 1
			print s
		elif aboveMedThreshold:
			#s = str(bestSim)
			s = "c\t"+sent2[i]+"\t"+correction
			output['corrections'] += 1
		else:
			#s = str(bestSim)
			s ="w\t"+sent2[i]+"\t"
			output['wrong'] += 1
			print s
		#else:
		#	s = str(bestSim)+"e\t"+sent2[i]+"\t"
		#	output['extras'] += 1
		#	print s
		i+=1
	i=0
	for k in missing:
		if k <= threshold:
			#s = str(k)
			s = "m\t"+sent1[i]+"\t"
			output['missing'] += 1
			print s
			
		i+=1
			
	# a correction is not considered missing or wrong
	output['missing'] -= output['corrections']

	return output

def runReportSimilarity(fileName,threshold=0.9):
        """ Assumes reports have FINDINGS: or REPORT: """
        fileText = [row.rstrip('\n') for row in open(fileName)]
                
        wordsToFind = ["FINDINGS:","REPORT:"]
        report1 = fileText[0]
        report2 = fileText[1]

        startLoc1 = -1
        #startLoc2 = -1
        for word in wordsToFind:
                if startLoc1 == -1 and report1.find(word) != -1:
                        startLoc1 = report1.find(word)+len(word)
                #if startLoc2 == -1 and report2.find(word) != -1:
                #       startLoc2 = report2.find(word)+len(word)

        report1 = report1[startLoc1:]
        #report2 = report2[startLoc2:]
        sentences1 = report1.split('.')
        sent1 = sentences1[:]
        sentences2 = report2.split('.')
        sent2 = sentences2[:]
                
        report1 = search_engine.textPreprocess(report1)
        report1 = search_engine.getDerivations(report1)
        report2 = search_engine.textPreprocess(report2)
        report2 = search_engine.getDerivations(report2)
        for i in range(len(sentences1)):
                sentences1[i] = search_engine.textPreprocess(sentences1[i])
                sentences1[i] = search_engine.getDerivations(sentences1[i])
        for i in range(len(sentences2)):
                sentences2[i] = search_engine.textPreprocess(sentences2[i])
                sentences2[i] = search_engine.getDerivations(sentences2[i])

        #corpus = gensim.corpora.MmCorpus('./model_files/reports_lsi.mm')
        tfidf_model = gensim.models.TfidfModel.load('./model_files/reports.tfidf_model')
        lsi_model = gensim.models.LsiModel.load('./model_files/reports.lsi_model')

        dictionary = gensim.corpora.Dictionary.load('./model_files/reports.dict')
        vec_lsi1 = lsi_model[tfidf_model[dictionary.doc2bow(report1)]]
        vec_lsi2 = lsi_model[tfidf_model[dictionary.doc2bow(report2)]]
        sen1Corp = [dictionary.doc2bow(sent) for sent in sentences1]
        sen2Corp = [dictionary.doc2bow(sent) for sent in sentences2]
        vec_lsis1 = lsi_model[tfidf_model[sen1Corp]]
        vec_lsis2 = lsi_model[tfidf_model[sen2Corp]]
        sCom = []

	# print corpus.num_terms
        # ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=corpus.num_terms)
        ind = gensim.similarities.MatrixSimilarity(vec_lsis1,num_features=10)
	# similarity table
        for  i in vec_lsis2:
                sCom.append(ind[i])
	missing = [0 for s in vec_lsis1]
	# obtain correct sentence
	i = 0

	output = {'missing': 0, 'corrections': 0, 'extras': 0, 'correct': 0}
	for col in sCom:
		aboveTopThreshold = False 
		#aboveMedThreshold = False
		j = 0
		correction = ""
		bestSim = 0
		for sim in col:
			if sim > threshold: 
				aboveTopThreshold = True 
			#elif sim > 0.5:
			else:
				correction = sent1[j]
			#	aboveMedThreshold = True
			if sim > bestSim:
				bestSim = sim
			if missing[j] < sim:
				missing[j] = sim
				
			j+=1
		if aboveTopThreshold: 
			#s = str(bestSim)
			s="n\t"+sent2[i]+"\t"	
			output['correct'] += 1
			print s
		#elif aboveMedThreshold:
		else:
			#s = str(bestSim)
			s ="c\t"+sent2[i]+"\t"+correction
			output['corrections'] += 1
			print s
		#else:
		#	s = str(bestSim)+"e\t"+sent2[i]+"\t"
		#	output['extras'] += 1
		#	print s
		i+=1
	i=0
	for k in missing:
		if k <= threshold:
			#s = str(k)
			s = "m\t"+sent1[i]+"\t"
			output['missing'] += 1
			print s
			
		i+=1
			
	return output
			

	'''
        # (1,2,3,...,n) for permutations of which sentence gets which sentence
        perms = []
        for i in range(len(sentences2)):
                if i < len(sentences1):
                        perms.append(i)
                else:
                        perms.append(-1)

        bestPerm = []
        bestPermResult = []
        bestResult = -1
        # iterate through all permutations and evaluate whether its a good fit
        for i in itertools.permutations(perms):
                nextResult = 0
                indice = 0
                permResult = [x for x in range(len(perms))]
                for j in i:
                        if j != -1:
                                nextResult += sCom[indice][j]
                                permResult[indice] = sCom[indice][j]
                        else:
                                permResult[indice] = -1
                        indice += 1
                if bestResult == -1 or bestResult < nextResult:
                        bestPerm = i
                        bestResult = nextResult
                        bestPermResult = permResult

        #print bestPerm 
	#put into necessary format and print to screen
        orderOfSent2 = []
        j=0
        for i in bestPerm:
                if i != -1:
                        orderOfSent2.append([sent2[j],sent1[i],bestPermResult[j]])
                        print sent2[j]+"\t"+sent1[i]+"\t"+str(bestPermResult[j])
                else:
                        orderOfSent2.append([sent2[j],"",-1])
                        print sent2[j]+"\t"+""+"\t"+str(-1)
                j+=1

        #print orderOfSent2     
        # evalutate which sentences are above the threshold, hence which sentences are right
        # which are wrong...
        index = gensim.similarities.MatrixSimilarity([vec_lsi1],num_features=corpus.num_terms)
        #print index
        sims = index[vec_lsi2]
        sims = sorted(enumerate(sims),key=lambda item: -item[1])
        #print sims
	'''


def getReportProcessed(report):
        wordsToFind = ["FINDINGS:","REPORT:"]
	# making sure the correct word statements exist to be able to 
	# process the report
	processable = False 
	for j in wordsToFind:		
		loc = report.find(j)
		if loc != -1: 
			processable = True
			report = report[loc+len(j):]
			
	if not processable:
		return ""
	else:
		return report

def reportsMissingPercentage(fileName,fileName2):
	random.seed()
	startTime = time.clock()
	numReportsToProcess = 10
	# open up the database and find a report
        fileText = [row.rstrip('\n') for row in open(fileName)]
	fileText2 = [row.rstrip('\n') for row in open(fileName)]

	# run for different thresholds
	thresholds = [x/100.0 for x in range(95,100,5)]
	totalTimes = 100
	print "Actual\tFound"
	for threshold in thresholds:
		numTimes = 0
		print "Current Threshold",threshold
		finalMissing = []	
		finalWrong = []
		for i in fileText:		
			orig = i[:]
			
			i = getReportProcessed(i)	
			if i == "":	
				continue

			sentences = i.split('.')[:-1]		
			senLen = float(len(sentences))

			# remove random number of sentences
			numberMissing = 0
			senLen = float(len(sentences))
			sentOrig = orig.split('.')[:-1]
			for sent in sentences:
				if random.random() < 0.5:		
					sentences.remove(sent)
					numberMissing += 1

			# insert random num sentence
			numberIncorrect = 0
			for i in range(len(sentences)):
				if random.random() < 0.5:
					# get random sentence to insert
					reportRandom = getReportProcessed(random.choice(fileText2))
					while reportRandom == "":
						reportRandom = getReportProcessed(random.choice(fileText2))
					sentenceRandom = random.choice(reportRandom.split('.')[:-1])
					sentences.append(sentenceRandom	)
					numberIncorrect += 1
					senLen += 1
			'''
			# swap random number of sentences
			numberIncorrect = 0
			cur = 0
			for sent in sentences:
				if random.random() < 0.5:
					# get random sentence to replace
					reportRandom = getReportProcessed(random.choice(fileText2))
					while reportRandom == "":
						reportRandom = getReportProcessed(random.choice(fileText2))
					sentenceRandom = random.choice(reportRandom.split('.')[:-1])
					sentences[cur] = sentenceRandom	
					numberIncorrect += 1
				cur += 1
			'''
			finalCopy = '.'.join(sentences)	
			orig = '.'.join(sentOrig)
			# save to file		
			with open('testingData','w') as f:				
				f.write(orig)
				f.write("\n")
				f.write(finalCopy)

			# record expected against received in file
			output = runReportSimilarity('testingData',threshold)		
			percentageWrong = 0
			percentageFoundWrong = 0
			percentageMissing = 0 
			percentageFoundMissing = 0 
			if len(sentences) != 0:
				percentageMissing = numberMissing / senLen
				percentageFoundMissing = output['missing'] / senLen
				percentageWrong = numberIncorrect / senLen
				percentageFoundWrong = output['corrections'] / senLen
			percentageMissing *= 100
			percentageFoundMissing *= 100
			percentageWrong *= 100
			percentageFoundWrong *= 100
			finalMissing.append([percentageMissing,percentageFoundMissing])
			finalWrong.append([percentageWrong,percentageFoundWrong])
			print percentageMissing, percentageFoundMissing
			print percentageWrong, percentageFoundWrong
			numTimes += 1
			if numTimes >= totalTimes:
				break;	
		with open('testForSimilarity/missing'+str(threshold)+'.csv','w') as f:
			for i in finalMissing:
				f.write(str(i[0])+","+str(i[1])+"\n")

		with open('testForSimilarity/wrong'+str(threshold)+'.csv','w') as f:
			for i in finalWrong:
				f.write(str(i[0])+","+str(i[1])+"\n")

	print "Total time", time.clock()-startTime

if __name__ == '__main__':
	if (len(sys.argv) < 2):
                print("ERROR: Please specify an input file")
                sys.exit()
        fileName = str(sys.argv[1])

	
       # runReportSimilarity(fileName)
	fileName2 = str(sys.argv[2])
	reportsMissingPercentage(fileName,fileName2)
