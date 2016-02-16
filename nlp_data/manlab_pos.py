import csv

def label(iteration):

    posarray = []
    posarray_labs = []
    uncertain = []

    pcount = 0
    ncount = 0
    ucount = 0

    directory = "it" + str(iteration) + "/"

    # Label 0 = negative, 1 = positive

    with open(directory + 'uncertain.csv','rb') as fin:
        try:
            for row in csv.reader(fin):
                uncertain.append(row[0])
        except:
            raise

    with open(directory + 'pos_reports.csv','rb') as fin:
        try:
            for row in csv.reader(fin):
                print ""
                print "REPORT:"
                print row[0]
                print ""
                while True:
                    answer = raw_input("Positive or negative? (Type p for positive or n for negative)")
                    if answer == "p":
                        posarray.append(row[0])
                        posarray_labs.append("POSITIVE")
                        pcount = pcount + 1
                        break
                    if answer == "n":
                        posarray.append(row[0])
                        posarray_labs.append("NEGATIVE")
                        ncount = ncount + 1
                        break
                    if answer == "u":
                        uncertain.append(row[0])
                        ucount = ucount + 1
                        break
                    else: 
                        print "Invalid input. Type p for positive, n for negative or u for uncertain."
                        inloop = 0
                print "Number of true positives: " + str(pcount)
                print "Number of false positives: " + str(ncount)
                print "Total uncertain reports: " + str(len(uncertain))

        except:
            raise

    with open(directory + 'pos_labelled.csv', 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        for i in range(len(posarray)):
            string = posarray_labs[i] + posarray[i]
            writer.writerow([string])

    with open(directory + 'uncertain.csv', 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        for i in range(len(uncertain)):
            string = uncertain[i]
            writer.writerow([string])

    print ""
    print "End of positive reports reached."
    print ""
