###########################

# This file removes all manually labelled reports from the unlabelled csv and puts them in the labelled csv.

###########################

import csv
import os
from shutil import copyfile

def swap(iteration,fileType):

    addreport_col1 = []
    addreport_col2 = []
    addreport_col3 = []

    report_stub = []
    report_lab = []
    count = 0
    labfound = 0

    labelled = "Cleaned" + fileType + "Labelled.csv"
    unlabelled = "Cleaned" + fileType + "Unlabelled.csv"

    directory = "it" + str(iteration) + "/"

    with open(directory + 'neg_labelled.csv','rb') as fin:
        try:
            for row in csv.reader(fin):
                count = count + 1
                rowstring = row[0]
                label = rowstring[0:8]
                if label == "POSITIVE":
                    reportstring = rowstring.partition("POSITIVE")
                    report_stub.append(reportstring[2])
                    report_lab.append("positive")
                elif label == "NEGATIVE":
                    reportstring = rowstring.partition("NEGATIVE")
                    report_stub.append(reportstring[2])
                    report_lab.append("negative")
                else:
                    print "ERROR"
                    print row[0]
                    break

        except:
            raise

    with open(directory + 'pos_labelled.csv','rb') as fin:
        try:
            for row in csv.reader(fin):
                count = count + 1
                rowstring = row[0]
                label = rowstring[0:8]
                if label == "POSITIVE":
                    reportstring = rowstring.partition("POSITIVE")
                    report_stub.append(reportstring[2])
                    report_lab.append("positive")
                elif label == "NEGATIVE":
                    reportstring = rowstring.partition("NEGATIVE")
                    report_stub.append(reportstring[2])
                    report_lab.append("negative")
                else:
                    print "ERROR"
                    print row[0]
                    break

        except:
            raise

    with open(unlabelled,'rb') as fin, open(directory + unlabelled, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        try:

            for row in csv.reader(fin):
                if not any (x in row[1] for x in report_stub):
                    col1 = row[0]
                    col2 = row[1]
                    string = [col1] + [col2]
                    writer.writerow(string)
                else:
                    labfound = labfound + 1
                    addreport_col1.append(row[0])
                    addreport_col2.append(row[1])
                    labno = -5000
   
                    # Find corresponding label
                    for i in range(len(report_stub)):
                        if report_stub[i] in row[1]:
                            labno = i
                    addreport_col3.append(report_lab[labno])

        except:
            raise

    print "Number of reports added to labelled reports: " + str(count)
    print "Number of reports removed from unlabelled reports: " + str(labfound)
    print ""

    with open(labelled,'rb') as fin, open(directory + labelled, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        try:

            for row in csv.reader(fin):
                col1 = row[0]
                col2 = row[1]
                col3 = row[2]
                string = [col1] + [col2] + [col3]
                writer.writerow(string)

        except:
            raise

        for i in range(len(addreport_col1)):
            col1 = addreport_col1[i]
            col2 = addreport_col2[i]
            col3 = addreport_col3[i]
            string = [col1] + [col2] + [col3]
            writer.writerow(string)

    # Remove old files from main directory
    os.remove(labelled)
    os.remove(unlabelled)

    # Copy new files into main directory
    os.system('cp ' + directory + labelled + " ../nlp_data/" + labelled)
    os.system('cp ' + directory + unlabelled + " ../nlp_data/" + unlabelled)
