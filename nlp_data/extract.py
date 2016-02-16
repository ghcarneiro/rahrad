import csv
import os

def extractFiles(iteration,fileType):
    directory = "it" + str(iteration) + "/"
    os.makedirs(directory)
    infile = "labelClassification.csv"

    # Names of output files
    outfile = "neg_reports.csv"
    outfile2 = "pos_reports.csv"

    otherimaging = ["Brains", "Pvab", "Plainab", "CTPA"]

    # Reports labelled negative by the classifier
    with open(infile,'rb') as fin, open(directory + outfile, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        imagingfound = 0
        found = 0

        for row in csv.reader(fin):
            try:
                if fileType in row[0] and len(row[0]) == len(fileType):
                    imagingfound = 1
                elif any (x in row[0] for x in otherimaging) and len(row[0]) <7:
                    imagingfound = 0
                elif "negative" in row[2] and imagingfound == 1:
                    found = 1
                elif found == 1:
                    string = row[0]
                    writer.writerow([string])
                    found = 0

            except:
                raise

    # Reports labelled positive by the classifier
    with open(infile,'rb') as fin, open(directory + outfile2, 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        imagingfound = 0
        found = 0

        for row in csv.reader(fin):
            try:
                if fileType in row[0] and len(row[0]) == len(fileType):
                    imagingfound = 1
                elif any (x in row[0] for x in otherimaging) and len(row[0]) <7:
                    imagingfound = 0
                elif "positive" in row[2] and imagingfound == 1:
                    found = 1
                elif found == 1:
                    string = row[0]
                    writer.writerow([string])
                    found = 0

            except:
                raise
    
    os.system('mv labelClassification.csv ' + directory + 'labelClassification.csv')
    os.system('mv coef.csv ' + directory + 'coef.csv')
