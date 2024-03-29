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

unique_words = []

# List of all the metaclasses in RadLex
metaclass_list = ["neuraxis_metaclass", "muscle_metaclass", "SetTermMetaclass", "immaterial_anatomic_metaclass", "anatomy_metaclass", "vein_metaclass", "artery_metaclass", "nerve_metaclass", "anatomical_space_metaclass", "obsolete_term_metaclass", "radlex_metaclass", "pathophysiology_metaclass", "bone_metaclass", "imaging_observation_metaclass", "CoordTermMetaclass", "DisjunctiveTermSetMetaclass", "ConjunctiveTermSetMetaclass"]

metaclass_start = []
metaclass_end = []

# Adding start and end tags to the metaclasses for search purposes
for x in range(len(metaclass_list)):
    start_string = "<" + metaclass_list[x]
    metaclass_start.append(start_string)
    end_string = "</" + metaclass_list[x]
    metaclass_end.append(end_string)

with open("radlex.owl","rb") as fin, open("radlex_unique_words_full.txt", "wb") as fout:

    found = 0
    pref_name = "ERROR"
    syn_name = "ERROR"

    for line in fin:
        try:

            # Find term in RadLex
            if any(word in line for word in metaclass_start):
                found = 1
                # print ("FOUND " + line) # DEBUG

            elif found == 1:

                # Extract preferred name
                if ("</Preferred_name" in line) or ("</Preferred_Name" in line):
                    nameSplit = line.split('>')
                    name = nameSplit[1].split('</Preferred')
                    pref_name = name[0].lower()
                    # print ("PREFERRED NAME FOUND " + line) # DEBUG

                # Extract any synonyms
                elif "</Synonym" in line:
                    nameSplit = line.split('>')
                    name = nameSplit[1].split('</Synonym')
                    if syn_name == "ERROR":
                        syn_name = name[0].lower()
                    else:
                        syn_name = syn_name + " " + name[0].lower()

                # End of term definition - print term name and synonyms to file
                elif any(word in line for word in metaclass_end):
                    pref_array = pref_name.split()

                    for x in range(len(pref_array)):
                        duplicate = 0
                        for y in range(len(unique_words)):
                            if pref_array[x] == unique_words[y]:
                                duplicate = 1
                        if duplicate == 0:
                            unique_words.append(pref_array[x])
                            string = pref_array[x]  + "\n"
                            fout.write(string)


                    if syn_name != "ERROR":
                        syn_array = syn_name.split()
                        for x in range(len(syn_array)):
                            duplicate = 0
                            for y in range(len(unique_words)):
                                if syn_array[x] == unique_words[y]:
                                    duplicate = 1
                            if duplicate == 0:
                                unique_words.append(syn_array[x])
                                string = syn_array[x] + "\n"
                                fout.write(string)


                    # Reset variables
                    found = 0
                    pref_name = "ERROR"
                    syn_name = "ERROR"
                    # print "VARIABLES RESET" # DEBUG

        except:
            print "ERROR"
