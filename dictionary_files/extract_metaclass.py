#
# ---------------------------------------------
# RADLEX METACLASS EXTRACTOR
# ---------------------------------------------
#
# Running this code will extract all the metaclasses from RadLex.
#

with open("radlex.owl","rb") as fin, open("metaclass_list.txt", "wb") as fout:

    meta_list = []
    found = 0

    for line in fin:
        try:
            if "etaclass rdf:ID" in line:
                metaclass_split = line.split('<')
                metaclass = metaclass_split[1].split(' rdf:ID')
                for x in range(len(meta_list)):
                    if metaclass[0] == meta_list[x]:
                        found = 1
                if found == 0:
                    meta_list.append(metaclass[0])
                found = 0

        except:
            print "ERROR"

    for x in range(len(meta_list)):
        string = '"' + meta_list[x] + '", '
        fout.write(string)
