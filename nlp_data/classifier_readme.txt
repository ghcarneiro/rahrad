NOTES

1. Copy all of the python files to your nlp_data folder. This should already contain the full, labelled and unlabelled reports from each corpus.
2. To change the parameters of the classifier, open the driver.py file. The iteration, fileType and threshold are the three parameters that can be changed.
3. You must manually change the number of the iteration every time the classifier is run. If you do not, it will get stuck on the extract.extractFiles(iteration,fileType) function. To get around this, comment out the previous functions in the driver file, change the iteration number and rerun driver.py. Remember to uncomment the previous functions afterwards!
4. If you want to record statistics such as true/false positives/negatives and the L2 norms, this must also be done manually. (The numbers are displayed in the terminal however.)
