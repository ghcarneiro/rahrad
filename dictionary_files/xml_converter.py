# ======================================
# XML CONVERTER
# ======================================
#
# Converts a text file containing a list of words into the XML format of the SPECIALIST lexicon.
#
# Updates
#
# 11/1/16 5:14pm - added UTF and <lexRecords> </lexRecords> tags
#

number = 900000 # Arbitary EUI number

with open("radlex_only_words.txt","rb") as fin, open("radlex_xml.xml", "wb") as fout:

    string = '<?xml version="1.0" encoding="UTF-8"?>\n<lexRecords>\n'
    fout.write(string)

    for line in fin:

        term = line.split('\n') # Remove newline
        eui = "E0" + str(number) # Generate unique EUI number

        # Generate XML code and write it to file
        string = "<lexRecord>\n\t<base>" + term[0] + "</base>\n\t<eui>" + eui + "</eui>\n\t<cat>noun</cat>\n"
        fout.write(string)
        string = '\t\t<inflVars cat="noun" cit="' + term[0] + '" eui="' + eui + '" infl="base" type="basic" unInfl="' + term[0] + '">' + term[0] + '</inflVars>\n'
        fout.write(string)
        string = '\t\t<inflVars cat="noun" cit="' + term[0] + '" eui="' + eui + '" infl="singular" type="basic" unInfl="' + term[0] + '">' + term[0] + '</inflVars>\n'
        fout.write(string)
        string = "\t\t<nounEntry>\n\t\t\t<variants>reg</variants>\n\t\t</nounEntry>\n</lexRecord>\n"
        fout.write(string)

        # Increment EUI number
        number = number + 1

    string = '</lexRecords>'
    fout.write(string)

