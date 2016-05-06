import sys

from sklearn.externals import joblib

usage = "USAGE: " + sys.argv[
    0] + " saved_model_file \n\t NOTE: use -i option on Python to interact with loaded model after report - (model loaded into 'rf')"
if len(sys.argv) != 2:
    print usage
    sys.exit(1)

model_file = sys.argv[1]
pipe = joblib.load(model_file)

rf = pipe.named_steps['classifier']

# There isn't much that can be discovered here other than the parameters that were defined
print "class_weight= " + str(rf.class_weight)
print "classes_= " + str(rf.classes_)
print "criterion= " + str(rf.criterion)
print "estimator_params= " + str(rf.estimator_params)
print "feature_importances_= " + str(rf.feature_importances_)
print "max_depth= " + str(rf.max_depth)
print "max_features= " + str(rf.max_features)
print "max_leaf_nodes= " + str(rf.max_leaf_nodes)
print "min_samples_leaf= " + str(rf.min_samples_leaf)
print "min_samples_split= " + str(rf.min_samples_split)
print "min_weight_fraction_leaf= " + str(rf.min_weight_fraction_leaf)
print "n_classes_= " + str(rf.n_classes_)
print "n_estimators= " + str(rf.n_estimators)
print "n_features_= " + str(rf.n_features_)
print "n_jobs= " + str(rf.n_jobs)
print "n_outputs_= " + str(rf.n_outputs_)
print "oob_score= " + str(rf.oob_score)
