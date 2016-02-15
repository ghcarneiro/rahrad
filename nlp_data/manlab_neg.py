import csv

def label(iteration):

    negarray = []
    negarray_labs = []
    uncertain = []

    pcount = 0
    ncount = 0
    ucount = 0

    directory = "it" + str(iteration) + "/"

    # Label 0 = negative, 1 = positive

    # Presents reports to be labelled
    with open(directory + 'neg_reports.csv','rb') as fin:
        try:
            for row in csv.reader(fin):
                print ""
                print "REPORT:"
                print row[0]
                print ""
                while True:
                    answer = raw_input("Positive or negative? (Type p for positive, n for negative or u for uncertain.)")
                    if answer == "p":
                        negarray.append(row[0])
                        negarray_labs.append("POSITIVE")
                        pcount = pcount + 1
                        break
                    if answer == "n":
                        negarray.append(row[0])
                        negarray_labs.append("NEGATIVE")
                        ncount = ncount + 1
                        break
                    if answer == "u":
                        uncertain.append(row[0])
                        ucount = ucount + 1
                        break
                    else: 
                        print "Invalid input. Type p for positive, n for negative or u for uncertain."
                        inloop = 0
                print "Number of true negatives: " + str(ncount)
                print "Number of false negatives: " + str(pcount)
                print "Uncertain reports: " + str(ucount)

        except:
            print "ERROR"

    # Write labelled reports to file
    with open(directory + 'neg_labelled.csv', 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        for i in range(len(negarray)):
            string = negarray_labs[i] + negarray[i]
            writer.writerow([string])

    # Write uncertain reports to a separate file
    with open(directory + 'uncertain.csv', 'wb') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        for i in range(len(uncertain)):
            string = uncertain[i]
            writer.writerow([string])

    print ""
    print "End of negative reports reached."
    print ""

