#
# ====================================================
# CSV REORDER
# ====================================================
#
# Description: Reorders the file of full reports to have the labelled reports first.
#

import csv
import os

def combine(iteration,fileType):

    labelled = []
    labelled_report = []
    full = []
    unlabelled = []
    unlabelled_report = []

    number = 0

    file1 = "Cleaned" + fileType + "Labelled.csv"
    file2 = "Cleaned" + fileType + "Unlabelled.csv"
    file3 = "Cleaned" + fileType + "Full.csv"

    directory = "it" + str(iteration) + "/"

    # Extract request numbers from labelled file
    with open(directory + file1,'rb') as fin:
        labelled_reader = csv.reader(fin, delimiter=",")

        try:
            for row in labelled_reader:
                if not "Req.No." in row[0]:
                    labelled.append(row[0]) 
                    labelled_report.append(row[1])  
   
        except:
            print "ERROR1"

    with open(directory + file2,'rb') as fin:
        unlabelled_reader = csv.reader(fin, delimiter=",")

        try:
            for row in unlabelled_reader:
                if not "Req.No." in row[0]:
                    unlabelled.append(row[0]) 
                    unlabelled_report.append(row[1])  
   
        except:
            print "ERROR2"

    print "Total labelled reports: " + str(len(labelled))
    print "Total unlabelled reports: " + str(len(unlabelled))
    print ""

    with open(directory + file3, 'wb') as fout:
        writer = csv.writer(fout)

        try:
            # Column headings
            col1 = "Req.No."
            col2 = "full_report"
            string = [col1] + [col2]
            writer.writerow(string)

            # Write all labelled reports to file
            for x in range(len(labelled)):
                col1 = labelled[x]
                col2 = labelled_report[x]
                string = [col1] + [col2]
                writer.writerow(string)
                # print string

            # Write all unlabelled reports to file:
            for x in range(len(unlabelled)):
                col1 = unlabelled[x]
                col2 = unlabelled_report[x]
                string = [col1] + [col2]
                writer.writerow(string)
                # print string
        except:
            print "ERROR3"

    os.remove(file3)

    # Copy new file into main directory
    os.system('cp ' + directory + file3 + " ../nlp_data/" + file3)

    print "Iteration complete. Change the iteration number in the driver.py file and run python driver.py to begin a new iteration."
