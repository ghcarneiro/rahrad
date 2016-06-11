from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn import svm

# Defines different combinations of feature extraction and classification in pipelines for easy use.

##########################
#### Model Parameters ####
##########################

count_vectorizer_params = {}
tfidf_vectorizer_params = {}
truncated_svd_params = {"n_components": 100}
random_forest_params = {'n_estimators': 500, 'min_samples_leaf': 3, 'max_depth': 10}
svc_params = {"probability": True}

#######################
#### Random Forest ####
#######################

# CountVectorizer -> LSI -> RandomForest
def get_count_lsi_randomforest():
    vectorizer = CountVectorizer(**count_vectorizer_params)
    svd = TruncatedSVD(**truncated_svd_params)
    classifier = RandomForestClassifier(**random_forest_params)

    return Pipeline([('vectorizer', vectorizer), ('lsi', svd), ('classifier', classifier)])


# TFIDFVectorizer -> LSI -> RandomForest
def get_tfidf_lsi_randomforest():
    vectorizer = TfidfVectorizer(**tfidf_vectorizer_params)
    svd = TruncatedSVD(**truncated_svd_params)
    classifier = RandomForestClassifier(**random_forest_params)

    return Pipeline([('vectorizer', vectorizer), ('lsi', svd), ('classifier', classifier)])

#############
#### SVM ####
#############

# CountVectorizer -> LSI -> SVM
def get_count_lsi_SVM():
    vectorizer = CountVectorizer(**count_vectorizer_params)
    svd = TruncatedSVD(**truncated_svd_params)
    classifier = svm.SVC(**svc_params)

    return Pipeline([('vectorizer', vectorizer), ('lsi', svd), ('classifier', classifier)])


# TFIDFVectorizer -> LSI -> SVM
def get_tfidf_lsi_SVM():
    vectorizer = TfidfVectorizer(**tfidf_vectorizer_params)
    svd = TruncatedSVD(**truncated_svd_params)
    classifier = svm.SVC(**svc_params)

    return Pipeline([('vectorizer', vectorizer), ('lsi', svd), ('classifier', classifier)])
