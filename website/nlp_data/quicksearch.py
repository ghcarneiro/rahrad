import csv

negatives = ["NO ", " NOT ", "UNLIKELY"]

number_array = []
number_reports = []

FILENAME = "CleanedPvabFull.csv"
searchterms = ["PERITONITIS"]
dx = "Acute abdomen"

with open(FILENAME,'rb') as fin:
    try:
        for row in csv.reader(fin):
            report = row[1]
            if "COMMENT:" in report:
                r = report.partition("COMMENT:")
	    elif "CONCLUSION:" in report:
                r = report.partition("CONCLUSION:")
	    elif "FINDINGS:" in report:
                r = report.partition("FINDINGS:")
	    elif "IMPRESSION:" in report:
                r = report.partition("IMPRESSION:")
	    elif "ASSESSMENT:" in report:
                r = report.partition("ASSESSMENT:")
	    else: 
		r = ["","",""]
            string = r[2].replace(",", ".")
            array = string.split(".")
	    for x in array:
                if all (z in x for z in searchterms):
                     if not any (y in x for y in negatives):
                         number_array.append(row[0])
                         number_reports.append(row[1])
	    
    except:
        raise

with open(searchterms[0] + '-reports.csv', 'wb') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    for i in range(len(number_array)):
        string = [number_array[i]] + [number_reports[i]] + [dx]
        writer.writerow(string)
