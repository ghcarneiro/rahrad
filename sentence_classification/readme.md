# [Sentence Classification](#sentence-classification)

This folder contains the components that make up the sentence classification component of the rahrad project. 
The goal of this module to explore and optimise the process of classifying sentences of a radiology report as **diagnostic** or not, as well as determining if the sentence represents a positive or negative diagnosis (aka **sentiment**).

**Jack Gerrits**

_Advanced Topics S1 2016_

## [Components](#components)
This project is split into several Python files with a specific purpose each. Most of the files are drivers which accept command line parameters in order to perform some function, then either write a file or display results.
### Drivers
#### Annotation Interface
The annotation driver is to provide a more user friendly and safe way to label the data files, ensuring that inputted values are valid. This program also utilises an active learning process to classify unlabelled sentences and serve up the sentences which the classifier is least confident about first, hopefully providing more information gain per label than it would be otherwise.

__annotation_driver.py__
```
USAGE: annotation_driver.py tag_type data_file
    tag_type = string('diagnostic'|'sentiment')
        Selects whether the diagnostic or sentiment labels will be tagged
    data_file = string
        The data file that the sentences will be read from and written back to with the respective tags
```

#### Automatic Learning
Automatic learning is the practice of automatically assigning labels to data, where the classifier is sufficiently confident. This file explores this process, it performs a number of passes over the data, training a classifier on each pass and selecting two sentences that it is most confident about in positive and negative tags and are also above a given threshold and applying the tags. It does this process until it has finished all passes and then writes the file. As well as actually applying these most confident tags it also keeps track of most confident tags in other buckets and at the end of the process provides a report of these tags for review. The buckets are (70,80), (80, 90) (90, 100]. 

__automatic_learning.py__
```
USAGE: automatic_learning.py type passes input_file output_file [saved_model]
    type = string('diagnostic'|'sentiment')
        Selects whether the diagnostic or sentiment labels will be tagged
    passes = integer
        How many times to apply a label to the most confident data
    input_file = string
        File that the data will be read from
    output_file = string
        File to write the resulting data to 
    saved_model = string
        File that contains the model parameters to be loaded into the pipeline.
        default - count_lsi_random_forest pipeline will be used
```

#### Model Generation
Model generation uses a labelled data set to optimise model parameters for it and saves the resulting model to json. The paramters to be evaluated by grid search are defined in the file in a dictionary as is a flag which defines whether random forest or SVM is being evaluated, so modifying this file is necessary to set the search parameters.
```
USAGE: model_generation.py input_file output_model_file
    input_file = string
        File that the data will be read from
    output_model_file = string
        File to write the resulting optimal model to in json format
```

#### Performance Evaluation
This driver facilitates testing a given model, or a default model. If a model is supplied, loads the parameters, otherwise default parameters are used. Splits data into training and testing sets, trains the model and evaluates the test set. Outputs a classification report as well as graphs of Recevier Operating Characteristics and Precision Recall for both training and testing.

__perf_eval.py__
```
USAGE: perf_eval.py type split_value data_file [saved_model]
    type = string('diagnostic'|'sentiment')
        Selects whether the diagnostic or sentiment labels should be tested
    split_value = float(0,1)
        Determines the split of training and test data. Value is proportion that is training.
    data_file = string
        File that the data will be read from
    saved_model = string
        File that contains the model parameters to be loaded into the pipeline.
        default - count_lsi_random_forest pipeline will be used
```
### Helpers
See `data_utils.py` and `pipelines.py` for helper function documentation.

#### Data class
This is the class that defines the structure of the report sentence object that are passed around, serialized and deserialized extensively throughout the project, contained in `data_utils.py`
```python
class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence        # string
        self.processed_sentence = ""    # string 
        self.diag_probs = []            # list(float)
        self.sent_probs = []            # list(float)
        self.diag_tag = ""              # string
        self.sent_tag = ""              # string
        self.report_id = ""             # string
        self.report_class = -1          # integer
        self.feature_vector = []        # list(float)
```

## [Data](#data-structure)
Data should be placed in a sub directory called `sentence_label_data` to keep it all in one place.
##### Core Data Files
The data that this project works with is stored in csv format, with each row being a single sentence and the columns being:
```
    sentence,processed sentence,diagnostic label,sentiment label,report id, report class
```

- _sentence_ - Original sentence extracted from the radiology report
- _processed sentence_ - This is the original sentence after it has undergone preprocessing, this involves:
    - Punctuation, single letter words and stop words are removed
    - Text is converted to lower case
    - Medical words are stemmed
- _diagnostic label_ - (p)ositive/(n)egative/(u)nsure - Value of the label assigned to this sentence for diagnostic
- _sentiment label_ - (p)ositive/(n)egative/(u)nsure - Value of the label assigned to this sentence for diagnostic
- _report id_ - ID of the original report that this sentence came from.
- _report class_ - Class or type of report that this sentence came from.

This is the most important datafile in this project and each row can be deserialized as a SentenceRecord object as in the `data_utils.read_from_csv` function. 
##### Raw report files
There are also csv files that contain raw reports, the above data files are generated from these report. The format for these reports is:
```
    report ID, full report
```

Using a database to store this information was considered and would likely be a better solution than simply storing csv files as rarely does the entire dataset need to be in memory at one time, would allow for better structure and more efficient queries.
However, csv was used as it is how the project had been structured before, because of the privacy implications of the data, easier to collaborate among colleagues and low cost of development.