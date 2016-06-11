import sys
from sklearn.metrics import classification_report, roc_curve, auc
import json
import data_utils
from sklearn.grid_search import GridSearchCV
import numpy as np
import pipelines

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "USAGE: " + sys.argv[0] + " input_file output_model_file"
        sys.exit(1)

    input_file = sys.argv[1]
    output_model_file = sys.argv[2]

    data = data_utils.read_from_csv(input_file)

    filtered_data = [x for x in data if x.diag_tag != "" and x.diag_tag != "u"]
    labels = [np.float32(x.diag_tag == "p") for x in filtered_data]
    data = [x.processed_sentence for x in filtered_data]
    report_ids = [x.report_id for x in filtered_data]

    train_data, train_labels, test_data, test_labels = data_utils.split_data(data, labels, report_ids, split=0.7)

    # change these parameters for the grid search
    # parameters = {'lsi__n_components': [100],
    #               'classifier__C': [3, 4, 5, 6, 7, 8, 9, 10],
    #               'classifier__kernel': ["rbf"]
    #               }

    parameters = {'lsi__n_components': [100],
                  'classifier__n_estimators': [1000],
                  'classifier__max_depth': [5, 10],
                  'classifier__min_samples_split': [5, 10],
                  'classifier__min_samples_leaf': [5, 10],
                  }

    # clf = GridSearchCV(pipelines.get_count_lsi_SVM(), parameters)
    clf = GridSearchCV(pipelines.get_count_lsi_randomforest(), parameters)
    clf.fit(train_data, train_labels)
    print "Best parameters set found o      n development set:"
    print ""
    print clf.best_params_
    print ""
    print "Grid scores on development set:"
    print ""
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
    print ""

    print "Detailed classification report:"
    print ""
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print ""
    y_true, y_pred = test_labels, clf.predict(test_data)
    print classification_report(y_true, y_pred)
    print ""

    json.dump(clf.best_params_, open(output_model_file, 'wb'))
