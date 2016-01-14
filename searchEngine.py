# process the search term
if (len(sys.argv) < 2):
    print("ERROR: Please specify an input file")
    sys.exit()
fileName = str(sys.argv[1])
fileText = [row.rstrip('\n') for row in open(fileName)]

if (fileText[1] == "notonlyreturndates"):
    print("Req No.<,>Report Date<,>Report")
elif (fileText[1] == "onlyreturndates"):
    print("Req No.<,>Report Date")
else:
    print("ERROR: input file layout error")
    sys.exit()

data = getData()

similarReports = search("lsi",50,fileText[0])
for reportIdx in similarReports:
    year = random.randint(2000,int(fileText[2][0:4])-1)
    month = random.randint(1,12)
    date = random.randint(1,28)
    if (fileText[1] == "notonlyreturndates"):
        print(data[reportIdx[0]][0] + "<,>" + str(year) + str(month).zfill(2) + str(date).zfill(2) + "<,>" + data[reportIdx[0]][1])
    elif (fileText[1] == "onlyreturndates"):
        print(data[reportIdx[0]][0] + "<,>" + str(year) + str(month).zfill(2) + str(date).zfill(2))
