from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# Defines different combinations of feature extraction and classification in pipelines for easy use.

#######################
#### Random Forest ####
#######################

# CountVectorizer -> RandomForest
def get_count_randomForest():
    vectorizer = CountVectorizer()
    forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=3)
    pipeline = Pipeline([('count', vectorizer), ('forest', forest)])
    return pipeline
#
# # TFIDFVectorizer -> RandomForest
# def get_tfidf_randomForest():
#
# # CountVectorizer -> LSI -> RandomForest
# def get_count_lsi_randomForest():
#     vectorizer = CountVectorizer()
#     svd = TruncatedSVD(n_components=10, random_state=42)
#
# # TFIDFVectorizer -> LSI -> RandomForest
# def get_tfidf_lsi_randomForest():
#
#
# #############
# #### SVM ####
# #############
# # CountVectorizer -> SVM
# def get_count_SVM():
#
# # TFIDFVectorizer -> SVM
# def get_tfidf_SVM():
#
# # CountVectorizer -> LSI -> SVM
# def get_count_lsi_SVM():
#
# # TFIDFVectorizer -> LSI -> SVM
# def get_tfidf_lsi_SVM():
#
#
# ##############################################
# #### Multilayer Perceptron Neural Network ####
# ##############################################
#
# # CountVectorizer -> Multilayer Perceptron Neural Network
# def get_count_MLP():
#
# # TFIDFVectorizer -> Multilayer Perceptron Neural Network
# def get_tfidf_MLP():
#
# # CountVectorizer -> LSI -> Multilayer Perceptron Neural Network
# def get_count_lsi_MLP():
#
# # TFIDFVectorizer -> LSI -> Multilayer Perceptron Neural Network
# def get_tfidf_lsi_MLP():