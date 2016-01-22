#
# ---------------------------------------------
# RADLEX TERM EXTRACTOR
# ---------------------------------------------
#
# For extracting terms from RadLex - preferred name and synonyms.
#
# The metaclass tags can be replaced with any other metaclass name to extract all terms from that metaclass.
#

with open("radlex.owl","rb") as fin, open("imagingtermlist.txt", "wb") as fout:

    found = 0
    rad_rid = "ERROR"
    pref_name = "ERROR"
    syn_name = "ERROR"

    for line in fin:
        try:

            # Find pathophysiology term in RadLex
            if "<imaging_observation_metaclass" in line:   # Exchange with "<pathophysiology_metaclass" to extract pathophysiology (disease process) terms instead
                found = 1
                # print ("FOUND " + line) # DEBUG

                # Extract RadLex ID
                ridSplit = line.split('rdf:ID="')
                rid = ridSplit[1].split('">')
                rad_rid = rid[0]
                # print "RID EXTRACTED" # DEBUG

            elif found == 1:

                # Extract preferred name
                if "</Preferred_name" in line:
                    nameSplit = line.split('>')
                    name = nameSplit[1].split('</Preferred')
                    pref_name = name[0]
                    # print ("PREFERRED NAME FOUND " + line) # DEBUG

                # Extract preferred name
                elif "</Synonym" in line:
                    nameSplit = line.split('>')
                    name = nameSplit[1].split('</Synonym')
                    if syn_name == "ERROR":
                        syn_name = name[0] + "\n"
                    else:
                        syn_name = syn_name + name[0] + "\n"

                # End of term definition - print term ID and name to file
                elif "</imaging_observation_metaclass>" in line:  # Replace with </pathophysiology_metaclass> to extrac pathophysiology terms instead
                    string = rad_rid + " " + pref_name + "\n"
                    fout.write(string)
                    if syn_name != "ERROR":
                        fout.write(syn_name)
                    # print "WRITTEN TO FILE" # DEBUG

                    # Reset variables
                    found = 0
                    rad_rid = "ERROR"
                    pref_name = "ERROR"
                    syn_name = "ERROR"
                    # print "VARIABLES RESET" # DEBUG

        except:
            print "ERROR"
