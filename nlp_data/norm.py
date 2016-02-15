import csv
import numpy as np

def calculateDifference(iteration):

    if iteration > 1:

        parameters = []
        old_parameters = []

        # Load previous iteration's parameters
        coef_directory = "it" + str(iteration - 1) + "/"
        with open(coef_directory + "coef.csv",'rb') as fin:
            for row in csv.reader(fin):
                try:
                    for i in range(10):
                        old_parameters.append(float(row[i]))
                except:
                    raise

        # Load current iteration's parameters
        with open("coef.csv",'rb') as fin:
            for row in csv.reader(fin):
                try:
                    for i in range(10):
                        parameters.append(float(row[i]))
                except:
                    raise

        # Convert lists to arrays
        current = np.asarray(parameters)
        old = np.asarray(old_parameters)

        print ""
        print "Current iteration's parameters:"
        print current
        print ""
        print "Previous iteration's parameters:"
        print old
        print ""
        print "Norm of the difference between both parameters: " + str(np.linalg.norm(current - old))
