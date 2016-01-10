#
# ---------------------------------------------
# RADLEX COMPLETE TERM EXTRACTOR
# ---------------------------------------------
#
# Running this code will generate a full list of terms contained in the RadLex dictionary. However, there will be three lines which say ERROR - 13450, 28043 and 32666. These can be removed.
#
# Line 13450 - There is no term missing here, unsure why it has printed ERROR
#
# Line 28043 - Non-english term only
#
# Line 32666 - No name specified


with open("LEXICON.xml","rb") as fin, open("specialist_full.txt", "wb") as fout:

    found = 0
    name_vars = []
    duplicate = 0


    for line in fin:
        try:

            # Find term in RadLex
            if "<lexRecord>" in line:
                found = 1
                # print ("FOUND " + line) # DEBUG

            elif found == 1:

                # Extract name
                if "<inflVars" in line:
                    nameSplit = line.split('>')
                    name = nameSplit[1].split('</inflVars')
                    lower_name = name[0].lower()
                     
                    for x in range(len(name_vars)):
                        if lower_name == name_vars[x]:
                            duplicate = 1

                    if duplicate == 0:
                        name_vars.append(lower_name)
                        string = lower_name + "\n"
                        fout.write(string)

                    duplicate = 0

                # End of term definition - print term name and synonyms to file
                elif "</lexRecord>" in line:
                    # Reset variables
                    found = 0
                    name_vars[:] = []
                    # print "VARIABLES RESET" # DEBUG

        except:
            print "ERROR"
