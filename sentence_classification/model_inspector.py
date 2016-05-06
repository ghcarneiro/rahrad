import sys

from sklearn.externals import joblib

usage = "USAGE: " + sys.argv[
    0] + " saved_model_file \n\t NOTE: use -i option on Python to interact with loaded model after report"
if len(sys.argv) != 2:
    print usage
    sys.exit(1)

model_file = sys.argv[1]
pipe = joblib.load(model_file)

rf = pipe.named_steps['classifier']
print "Number of trees: " + str(len(rf.estimators_))
# embed()
# print [estimator.tree_.depth for estimator in rf.estimators_]
# print "Average tree depth: " + numpy.mean([estimator.tree_.max_depth for estimator in rf.estimators_])


def get_code(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    print tree.tree_.feature
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node):
        if (threshold[node] != -2):
            print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node])
            print "} else {"
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node])
            print "}"
        else:
            print "return " + str(value[node])

    recurse(left, right, threshold, features, 0)

get_code(rf.estimators_[0], ['no', 'yes'])
