import json
import sys
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pipelines
import data_utils
import numpy as np

# Change this variable to false to use svm
USE_RF = True
# Target class for report class classification so that it can be a binary task
TARGET_CLASS = 3

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print "USAGE: " + sys.argv[0] + " type split_value data_file [saved_model]"
        sys.exit(1)

    test_type = sys.argv[1]
    split_value = float(sys.argv[2])
    data_file = sys.argv[3]
    model_params = dict()

    if len(sys.argv) == 5:
        prebuilt_model_file = sys.argv[4]
        model_params = json.load(open(prebuilt_model_file, 'rb'))
        # ensures this field is a non unicode string
        if 'classifier__kernel' in model_params:
            model_params['classifier__kernel'] = str(model_params['classifier__kernel'])

    if test_type == 'diagnostic':
        tag_attr = 'diag_tag'
        TARGET_POSITIVE = 'p'
    elif test_type == 'sentiment':
        tag_attr = 'sent_tag'
        TARGET_POSITIVE = 'p'
    elif test_type == 'class':
        tag_attr = 'report_class'
        TARGET_POSITIVE = TARGET_CLASS
    else:
        raise ValueError("Unknown tag: " + test_type)

    data = data_utils.read_from_csv(data_file)
    filtered_data = [x for x in data if getattr(x, tag_attr) != "" and getattr(x, tag_attr) != "u"]
    filtered_data = filtered_data[:2500]  # put a limit on the size for performance

    labels = [np.float32(getattr(x, tag_attr) == TARGET_POSITIVE) for x in filtered_data]
    report_ids = [x.report_id for x in filtered_data]
    sentences = [x.processed_sentence for x in filtered_data]

    print "0:" + str(len([x for x in filtered_data if getattr(x, tag_attr) == 0]))
    print "1:" + str(len([x for x in filtered_data if getattr(x, tag_attr) == 1]))
    print "2:" + str(len([x for x in filtered_data if getattr(x, tag_attr) == 2]))
    print "3:" + str(len([x for x in filtered_data if getattr(x, tag_attr) == 3]))

    train_data, train_labels, test_data, test_labels = data_utils.split_data(sentences, labels, report_ids, split_value)

    # Create transformation pipeline
    if USE_RF:
        pipe = pipelines.get_count_lsi_randomforest()
    else:
        pipe = pipelines.get_count_lsi_SVM()

    # set pipe parameters and train model
    pipe.set_params(**model_params)
    pipe.fit(train_data, train_labels)

    print "Total = " + str(len(filtered_data)) + " [" + str(labels.count(0)) + ", " + str(labels.count(1)) + "]"
    print "Train = " + str(len(train_data)) + " [" + str(train_labels.count(0)) + ", " + str(train_labels.count(1)) + "]"
    print "Test = " + str(len(test_data)) + " [" + str(test_labels.count(0)) + ", " + str(test_labels.count(1)) + "]"

    # Training performance data
    y_true_train = train_labels
    y_pos_score_train = [pipe.predict_proba([x]).tolist()[0][1] for x in train_data]

    # Testing performance data
    y_true_test = test_labels
    y_pred_test = [pipe.predict([x]).tolist()[0] for x in test_data]
    y_pos_score_test = [pipe.predict_proba([x]).tolist()[0][1] for x in test_data]
    print classification_report(y_true_test, y_pred_test, target_names=['yes', 'no'])

    # Plot both sets of performance curves
    plt.figure()
    data_utils.add_curves("Train", y_true_train, y_pos_score_train, 1)
    data_utils.add_curves("Test", y_true_test, y_pos_score_test, 3)

    print "Failed prediction report:"
    for i in xrange(len(test_data)):
        if y_true_test[i] != y_pred_test[i]:
            print test_data[i]
            print "Actual: " + str(y_true_test[i])
            print "Predicted: " + str(y_pred_test[i])
            print "(" + str(1 - y_pos_score_test[i]) + ", " + str(y_pos_score_test[i]) + ")"
            print ""

    plt.show()
