radlex = []
specialist = []

with open('radlex_full.txt','rb') as fin:
    try:
        for line in fin:
            radlex.append(line)

    except:
        print "ERROR"

with open('specialist_full.txt','rb') as fin:
    try:
        for line in fin:
            specialist.append(line)

    except:
        print "ERROR"

# Generate list of shared terms
shared_terms = list(set(radlex).intersection(specialist))

with open('shared_terms.txt', 'wb') as fout:
    for i in range(len(shared_terms)):
        fout.write(shared_terms[i])

for x in shared_terms:
    radlex.remove(x)
    specialist.remove(x)

with open('radlex_only.txt', 'wb') as fout:
    for i in range(len(radlex)):
        fout.write(radlex[i])

with open('specialist_only.txt', 'wb') as fout:
    for i in range(len(specialist)):
        fout.write(specialist[i])
