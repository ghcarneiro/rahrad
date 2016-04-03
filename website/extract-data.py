old_level1 = "0"
level1 = "0"
old_level2 = "0"
level2 = "0"
old_level3 = "0"
level3 = "0"
diagnosis = "0"
current_category = "0"
linecount = 0

with open("diagnosis-list.txt","rb") as fin, open("test-structure.txt", "wb") as fout:
    for line in fin:
        try:
		# Extract levels
		if "LEVEL 1:" in line:
			string = line
			level1_slice = string.partition("1: ")
			remove_newline = level1_slice[2].partition('\r')
			old_level1 = level1
			level1 = (remove_newline[0].lower()).capitalize()
		elif "LEVEL 2:" in line:
			string = line
			level2_slice = string.partition("2: ")
			remove_newline = level2_slice[2].partition('\r')
			old_level2 = level2
			string = (remove_newline[0].lower()).capitalize()
			if (string != level2) and (level3 == "None"):
				concat = level1 + " -> " + level2 + "\n"
				fout.write(concat)
			level2 = string
		elif "LEVEL 3:" in line:
			string = line
			level3_slice = string.partition("3: ")
			remove_newline = level3_slice[2].partition('\r')
			string = (remove_newline[0].lower()).capitalize()
			if (string != level3) and (level3 != "None") and (level3 != "0"):
				concat = old_level1 + " -> " + old_level2 + " -> " + level3 + "\n"
				fout.write(concat)
			level3 = string
				

		# Error checking has been implemented to ensure that no categories have been missed
		elif "CATEGORY 1" in line:
			if (current_category == "0") or (current_category == "3"):
				current_category = "1"
			else:
				print "ERROR " + str(linecount)
		elif "CATEGORY 2" in line:
			if (current_category == "1"):
				current_category = "2"
			else:
				print "ERROR " + str(linecount)
		elif "CATEGORY 3" in line:
			if (current_category == "2"):
				current_category = "3"
			else:
				print "ERROR " + str(linecount)

		linecount = linecount + 1

			
	except:
		raise
