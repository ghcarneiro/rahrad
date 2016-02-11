# driver.py
import preprocess
import modelGeneration
import search
import generateReports
import rnn
# preprocess.buildMedDict()
# preprocess.preprocessReports()
# modelGeneration.buildDictionary()
# modelGeneration.buildModels()
# modelGeneration.buildWord2VecModel()
# modelGeneration.buildDoc2VecModel()
# modelGeneration.testClassification()
# searchTerm = "haemorrhage"
# searchTerm = "no haemorrhage"
# searchTerm = "left sided embolus"
# searchTerm = "2400      CT HEAD - PLAIN L3  CT HEAD:  CLINICAL DETAILS:  INVOLVED IN FIGHT, KICKED IN HIS HEAD, VOMITED AFTER THIS WITH EPISODIC STARING EPISODES WITH TEETH GRINDING. ALSO INTOXICATED (BREATH ALCOHOL ONLY 0.06). PROCEDURE:  PLAIN SCANS THROUGH THE BRAIN FROM SKULL BASE TO NEAR VERTEX. IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  REPORT:  VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE AND IT IS SYMMETRICAL AROUND THE MIDLINE.  NORMAL GREY/WHITE DIFFERENTIATION.  NO INTRACEREBRAL HAEMATOMA OR EXTRA AXIAL COLLECTION. NO CRANIAL VAULT FRACTURE SEEN.  COMMENT: STUDY WITHIN NORMAL LIMITS."
# searchTerm = "GREY/WHITE MATTER DIFFERENTIATION"
# search.searchEngineTest("rnn",searchTerm)
generateReports.precisionRecall("pr_tests.csv")
# generateReports.labelClassification()
# generateReports.labelClassificationD2V()

# rnn.preprocessReports()
# rnn.buildWord2VecSentences()
# rnn.testWord2VecModel()

# rnn.buildSentenceRNN(epochs=15)
# rnn.buildSentenceRNN(epochs=20,continueTraining=True)
# rnn.sentenceToEncoder()
# rnn.sentencesToDense()

# rnn.searchRNN("haemorrhage")
# rnn.searchRNN("no haemorrhage")
# rnn.searchRNN("left sided embolus")

# rnn.compareSentences("There is a intracranial haemorrhage","There is a haemorrhage in the cranium")
# rnn.compareSentences("There is no intracranial haemorrhage","There is a haemorrhage in the cranium")
# rnn.compareSentences("There is a intracranial haemorrhage","The study is within normal limits")
# rnn.compareSentences("There is a intracranial haemorrhage.","There is a haemorrhage in the cranium.")
# rnn.compareSentences("There is no intracranial haemorrhage.","There is a haemorrhage in the cranium.")
# rnn.compareSentences("There is a intracranial haemorrhage.","The study is within normal limits.")

# rnn.nextWords("VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE AND IT IS")
# rnn.nextWords("VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE")
# rnn.nextWords("NO INTRACEREBRAL HAEMATOMA OR")
# rnn.nextWords("left sided embolus")


# rnn.reportsToDense()
# rnn.buildReportRNN(epochs=50)
# rnn.buildReportRNN(epochs=10,continueTraining=True)
# rnn.reportToEncoder()
# rnn.reports2vecs()
