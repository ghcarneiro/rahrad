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
# searchTerm = "2400      CT HEAD - PLAIN L3  CT HEAD:  CLINICAL DETAILS:  INVOLVED IN FIGHT, KICKED IN HIS HEAD, VOMITED AFTER THIS WITH EPISODIC STARING EPISODES WITH TEETH GRINDING. ALSO INTOXICATED (BREATH ALCOHOL ONLY 0.06). PROCEDURE:  PLAIN SCANS THROUGH THE BRAIN FROM SKULL BASE TO NEAR VERTEX. IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  REPORT:  VENTRICULAR CALIBRE IS WITHIN NORMAL LIMITS FOR AGE AND IT IS SYMMETRICAL AROUND THE MIDLINE.  NORMAL GREY/WHITE DIFFERENTIATION.  NO INTRACEREBRAL HAEMATOMA OR EXTRA AXIAL COLLECTION. NO CRANIAL VAULT FRACTURE SEEN.  COMMENT: STUDY WITHIN NORMAL LIMITS."
# searchTerm = "GREY/WHITE MATTER DIFFERENTIATION"
# search.searchEngineTest("rnn",searchTerm)
# generateReports.precisionRecall("pr_tests.csv")
# generateReports.labelClassification()
# generateReports.labelClassificationD2V()

rnn.preprocessReports()
rnn.buildWord2Vec()
# rnn.buildRNN()
# rnn.fullToEncoder()
# rnn.buildPredictionsRNN()
rnn.buildSentenceRNN()
