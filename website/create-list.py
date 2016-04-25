old_level1 = "0"
level1 = "0"
level1_count = 0
old_level2 = "0"
level2 = "0"
level2_count = 0
old_level3 = "0"
level3 = "0"
level3_count = 0

diagnosis = "0"
current_category = "0"
linecount = 0

with open("diagnosis-list.txt","rb") as fin, open("conceptlist.txt", "wb") as fout:
    for line in fin:
        try:
		# Extract levels
		if "LEVEL 1:" in line:
			string = line
			level1_slice = string.partition("1: ")
			remove_newline = level1_slice[2].partition('\r')
			old_level1 = level1
			string = (remove_newline[0].lower()).capitalize()
			if (string != level1):
				text = "L1_" + string + "_0\n"
				fout.write(text)
				level1_count = level1_count + 1
				level2_count = 0
			level1 = string
		elif "LEVEL 2:" in line:
			string = line
			level2_slice = string.partition("2: ")
			remove_newline = level2_slice[2].partition('\r')
			old_level2 = level2
			string = (remove_newline[0].lower()).capitalize()
			if (string != level2):
				text = "L2_" + string + "_0\n"
				fout.write(text)
				level2_count = level2_count + 1
				level3_count = 0
			level2 = string
		elif "LEVEL 3:" in line:
			string = line
			level3_slice = string.partition("3: ")
			remove_newline = level3_slice[2].partition('\r')
			string = (remove_newline[0].lower()).capitalize()
			if (string != level3) and (string != "None"):
				text = "L3_" + string + "_0\n"
				fout.write(text)
				level3_count = level3_count + 1
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

		else:
			line = line.strip()
			if (line != "") and (line != "NONE"):
				diagnosis = line.decode('utf-8', 'ignore').encode('utf-8') 
				text = "E_" + diagnosis + "_0\n"
				fout.write(text)

		linecount = linecount + 1

			
	except:
		raise
