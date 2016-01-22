There are 4 diagnoses/data sets:

Brains (labelled diagnosis is haemorrhage)
CTPA (diagnosis is pulmonary embolism)
Plainab (diagnosis is ureteric calculus)
Pvab (diagnosis is diverticulitis)

Each data set has three files associated. “Full” is the entire corpus. “Labelled” is 80-90 labelled cases for that diagnosis (I will probably need to do more, but see how you go). “Unlabelled” is “Full” – “Labelled”. 
The unlabelled cases are not negative examples, they contain both positives and negatives.

The “Full” and “Unlabelled” data sets are 2 column tables. 
The “Labelled” data sets are 3 column tables (each has an extra column for the labels). 

Each column has a header (for example, “full_report” or “haemorrhage”).

The index of each table is the unique accession number in the “Req. No.” column.

Each report text is in the “full_report” column, and is presented as a single long string. It is capitalised, and contains all punctuation. I haven’t removed stopwords, or done stemming.

I hope that all makes sense.
