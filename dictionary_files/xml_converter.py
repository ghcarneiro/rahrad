# ======================================
# XML CONVERTER
# ======================================
#
# Converts a text file containing a list of words into the XML format of the SPECIALIST lexicon.
#

number = 900000 # Arbitary EUI number

with open("radlex_only_words.txt","rb") as fin, open("radlex_xml.xml", "wb") as fout:
    for line in fin:

        term = line.split('\n') # Remove newline
        eui = "E0" + str(number) # Generate unique EUI number

        # Generate XML code and write it to file
        string = "<lexRecord>\n    <base>" + term[0] + "</base>\n    <eui>" + eui + "</eui>\n    <cat>noun</cat>\n"
        fout.write(string)
        string = '        <inflVars cat="noun" cit="' + term[0] + '" eui="' + eui + '" infl="base" type="basic" unInfl="' + term[0] + '">' + term[0] + '</inflVars>\n'
        fout.write(string)
        string = '        <inflVars cat="noun" cit="' + term[0] + '" eui="' + eui + '" infl="singular" type="basic" unInfl="' + term[0] + '">' + term[0] + '</inflVars>\n'
        fout.write(string)
        string = "        <nounEntry>\n            <variants>reg</variants>\n        </nounEntry>\n</lexRecord>\n"
        fout.write(string)

        # Increment EUI number
        number = number + 1

