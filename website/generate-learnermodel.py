level1 = "0"
level2 = "0"
level3 = "0"
diagnosis = "0"
current_category = "0"
linecount = 0

with open("diagnosis-list.txt","rb") as fin, open("test-learner.txt", "wb") as fout:
    for line in fin:
        try:
		# Extract levels
		if "LEVEL 1:" in line:
			string = line
			level1_slice = string.partition("1: ")
			remove_newline = level1_slice[2].partition('\r')
			level1 = (remove_newline[0].lower()).capitalize()
		elif "LEVEL 2:" in line:
			string = line
			level2_slice = string.partition("2: ")
			remove_newline = level2_slice[2].partition('\r')
			level2 = (remove_newline[0].lower()).capitalize()
		elif "LEVEL 3:" in line:
			string = line
			level3_slice = string.partition("3: ")
			remove_newline = level3_slice[2].partition('\r')
			level3 = (remove_newline[0].lower()).capitalize()

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
				string = "LearnerDx.create(user: 'teststudent', diagnosis: '" + diagnosis + "', last_attempt: 0, review_list: no, cases_attempted: 0, cases_correct: 0, cases_excellent: 0) \n"
				fout.write(string)

		linecount = linecount + 1

			
	except:
		raise
