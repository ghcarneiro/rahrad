import preprocess
import modelGeneration
import norm
import extract
import manlab_neg as neg
import manlab_pos as pos
import swapreports
import reorder

############################################################################################################

# CLASSIFIER PARAMETERS

# These three parameters may be altered as required

# Change this for each iteration (otherwise it won't work!)
iteration = 1

fileType = "Plainab" # Valid fileTypes are Brains, CTPA, Plainab and Pvab
threshold = 0.001 # Reports below this threshold of uncertainty will be presented for manual labelling

############################################################################################################

# Leon/William's code
preprocess.preprocessReports(fileType)
modelGeneration.buildDictionary(fileType)
modelGeneration.buildModels()
modelGeneration.testClassification(threshold,fileType)

# Calculates the norm of the difference between the current model's parameters and the previous model's parameters
norm.calculateDifference(iteration) 

# For extraction and labelling of data
extract.extractFiles(iteration,fileType) # Extracts files to be labelled, separated by positive and negative classification
neg.label(iteration) # Manually label reports classified as negative
pos.label(iteration) # Manually label reports classified as positive
swapreports.swap(iteration,fileType) # Move newly labelled reports from unlabelled to labelled file
reorder.combine(iteration,fileType) # Combine labelled and unlabelled reports
