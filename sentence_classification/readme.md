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
            The data file that the sentences will be read from adn written back to with the respective tags
```

#### Automatic Learning
Automatic learning is the practice of automatically assigning labels to data, where the classifier is sufficiently confident. This file explores this process, it performs a number of passes over the data, training a classifier on each pass and selecting two sentences that it is most confident about in positive and negative tags and are also above a given threshold and applying the tags. It does this process until it has finished all passes and then writes the file. As well as actually applying these most confident tags it also keeps track of most confident tags in other buckets and at the end of the process provides a report of these tags for review. The buckets are (70,80), (80, 90) (90, 100]. 
__automatic_learning.py__
```
    USAGE: automatic_learning.py type passes input_file output_file
        type = string('diagnostic'|'sentiment')
            Selects whether the diagnostic or sentiment labels will be tagged
        passes = integer
            How many times to apply a label to the most confident data
        input_file = string
            File that the data will be read from
        output_file = string
            File to write the resulting data to 
```

#### Model Generation
__In progress__
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
#### Data Utils
This file defines functions that generally operate on the data to be used in the driver code.
```
class SentenceRecord(object):
    def __init__(self, sentence):
        self.sentence = sentence        # string
        self.processed_sentence = ""    # string 
        self.diag_probs = []            # list(float)
        self.sent_probs = []            # list(float)
        self.diag_tag = ""              # string
        self.sent_tag = ""              # string
        self.report_id = ""             # string
```
Main data class that the sentences are read and written with and passed around.

```
def write_to_csv(sentence_file, sentence_tags):
    sentence_file: string
        File path to write data to
    sentence_tags: list(SentenceRecord))
        List of SentenceRecord objects to write to the given filepath.
    
    return None
        Function does not return anything
```
Writes the given list of SentenceRecord files to the given file.

```
def read_from_csv(sentence_file{string}):
    sentence_file: string
        File path to read from.

    return list(SentenceRecord)
        Returns the list of Sentence Record objects that were read from disk.
```
Reads a list of SentenceRecord objects from the given datafile.

```
def generate_sentences(data_files):
   
    return data
```
```
def remove_duplicates(data):
   
    return res
```
```
def strip_labels(data):
    
    return data
```
```
def generate_sentences_from_raw():
        output_file = './sentence_label_data/sentences_ALL.csv'
```

```
def generate_sentence_id_dict():
  
    return sentence_ids
```
```
def add_sentence_report_ids(data_file, sentence_id_file='./sentence_label_data/sentence_id_mappings.csv'):
    csv.field_size_limit(sys.maxsize)

    write_to_csv(data_file, labelled_data)
```
```
def split_data(data, labels, report_ids, split=0.5, shuffle_items=True):
    
    return train_data, train_labels, test_data, test_labels
```
#### Pipelines
## [Data](#data-structure)
Data should be placed in a sub directory called `sentence_label_data` to keep it all in one place.
##### Core Data Files
The data that this project works with is stored in csv format, with each row being a single sentence and the columns being:
```
    sentence,processed sentence,diagnostic label,sentiment label,report id
```
- _sentence_ - Original sentence extracted from the radiology report
- _processed sentence_ - This is the original sentence after it has undergone preprocessing, this involves:
    - Punctuation, single letter words and stop words are removed
    - Text is converted to lower case
    - Medical words are stemmed
- _diagnostic label_ - (p)ositive/(n)egative/(u)nsure - Value of the label assigned to this sentence for diagnostic
- _sentiment label_ - (p)ositive/(n)egative/(u)nsure - Value of the label assigned to this sentence for diagnostic
- _report id_ - ID of the original report that this sentence came from.

This is the most important datafile in this project.
##### Raw report files
There are also csv files that contain raw reports, the above data files are generated from these reports: The format for these reports is
```
    report ID, full report
```

##### Report ID Mappings
In order to convert pre-report ID data to include report IDs the `generate_sentence_id_dict` and `add_sentence_report_ids` functions can be used.
The former outputs a dictionary of `sentence => report ID` mappings which can be serialised to save time for future uses. Since it is just a dictionary the format is:
```
    sentences, list of report IDs
```

Using a database to store this information was considered and would likely be a better solution than simply storing csv files as rarely does the entire dataset need to be in memory at one time, would allow for better structure and more efficient queries.
However, csv was used as it is how the project had been structured before, because of the privacy implications of the data, easier to collaborate among colleagues and low cost of development.