# Sentence Classification

This folder contains the components that make up the sentence classification component of the rahrad project. 
The goal of this module to explore and optimise the process of classifying sentences of a radiology report as **diagnostic** or not, as well as determining if the sentence represents a positive or negative diagnosis (aka **sentiment**).

**Jack Gerrits**
_Advanced Topics S1 2016_

### Components

### Data
Data is structured as csv files with the format being:
```
    sentence,processed sentence,diagnostic label,sentiment label,report id
```
- _sentence_ - Original sentence extracted from the radiology report
- _processed sentence_ - This is the original sentence after it has undergone preprocessing, this involves:
    - stage 1
- _diagnostic label_
- _sentiment label_
- _report id_

Data should be placed in a sub directory called `sentence_label_data`.
Many of the files and functions define and input and output file.