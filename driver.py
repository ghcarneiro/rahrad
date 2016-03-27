# driver.py
import preprocess
import modelGeneration
import search
import generateReports
import rnn
#preprocess.buildMedDict()
#preprocess.preprocessReports()
#modelGeneration.buildDictionary()
#modelGeneration.buildModels()
#modelGeneration.buildWord2VecModel()
#modelGeneration.buildDoc2VecModel()
#modelGeneration.testClassification()
# searchTerm = "haemorrhage"
# searchTerm = "no haemorrhage"
# searchTerm = "left sided embolus"
# searchTerm = "2400      CT HEAD - PLAIN L3  CT HEAD:  CLINICAL DETAILS:  INVOLVED IN FIGHT, KICKED IN HIS HEAD, VOMITED AFTER THIS WITH EPISODIC STARING EPISODES WITH TEETH GRINDING. ALSO INTOXICATED (BREATH ALCOHOL ONLY 0.06). PROCEDURE:  PLAIN SCANS THROUGH THE BRAIN FROM SKULL BASE TO NEAR VERTEX. IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  REPORT:  VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE AND IT IS SYMMETRICAL AROUND THE MIDLINE.  NORMAL GREY/WHITE DIFFERENTIATION.  NO INTRACEREBRAL HAEMATOMA OR EXTRA AXIAL COLLECTION. NO CRANIAL VAULT FRACTURE SEEN.  COMMENT: STUDY WITHIN NORMAL LIMITS."
# searchTerm = "GREY/WHITE MATTER DIFFERENTIATION"
# search.searchEngineTest("rnn",searchTerm)
# generateReports.precisionRecall("pr_tests.csv")
# generateReports.labelClassification()
# generateReports.labelClassificationD2V()

# # Train the RNN sentence model using the small dataset
#rnn.preprocessReports()
#rnn.buildWord2VecSentences()
rnn.buildSentenceRNN(epochs=10)
rnn.sentenceToEncoder()

# # Train the RNN sentence model using the full dataset
# rnn.preprocessFullReports()
# rnn.buildWord2VecSentences()
# rnn.buildSentenceRNN(epochs=10)
# rnn.sentenceToEncoder()

# # Test the word2vec model that is used in the RNN sentence model
# rnn.testWord2VecModel()

# # Train the sentence RNN model for additional epochs
# rnn.buildSentenceRNN(epochs=6,continueTraining=True)

# # Use the RNN model to search for sentences
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

# # Build the RNN Report model using the RNN sentence model
# rnn.reportsToDense()
# rnn.buildReportRNN(epochs=20)
# rnn.reportToEncoder()
# rnn.reports2vecs()

# # Train the RNN report model for additional epochs
# rnn.buildReportRNN(epochs=20,continueTraining=True)

# generateReports.labelClassificationRNN()
# generateReports.labelClassificationRNN(learn=False)

# # Test the RNN model report comparison tool
# print("loading reports")
# reports = preprocess.getReports()
# print("loaded reports")
# print("report 1:")
# print(reports[300])
# print("report 2:")
# print(reports[3000])
# print(rnn.compareReportSentences(reports[300],reports[3000]))
