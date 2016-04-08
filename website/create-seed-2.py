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

with open("diagnosis-list.txt","rb") as fin, open("test-seed.txt", "wb") as fout:
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
				concat = 'd1_' + str(level1_count) + ' = ' + 'DxLevel1.create(name: "' + string + '")\n' 
				fout.write(concat)
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
				concat = '	d1_' + str(level1_count -1) + '_d2_' + str(level2_count) + ' = ' + 'DxLevel2.create(name: "' + string + '", dx_level1_id: d1_' + str(level1_count -1) + '.id)\n' 
				fout.write(concat)
				level2_count = level2_count + 1
				level3_count = 0
			level2 = string
		elif "LEVEL 3:" in line:
			string = line
			level3_slice = string.partition("3: ")
			remove_newline = level3_slice[2].partition('\r')
			string = (remove_newline[0].lower()).capitalize()
			if (string != level3) and (string != "None"):
				concat = '		d1_' + str(level1_count -1) + '_d2_' + str(level2_count - 1) + '_d3_' + str(level3_count) + ' = ' + 'DxLevel3.create(name: "' + string + '", dx_level2_id: d1_' + str(level1_count -1) + '_d2_' + str(level2_count -1) + '.id)\n'
				fout.write(concat)
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
				if (level3 == "None"):
					string = '			EndDx.create(name: "' + diagnosis + '", category: "' + current_category + '", dx_level2_id: d1_' + str(level1_count -1) + '_d2_' + str(level2_count -1) + '.id, frequency: 0.1) \n'
				else:
					string = '			EndDx.create(name: "' + diagnosis + '", category: "' + current_category + '", dx_level3_id: d1_' + str(level1_count -1) + '_d2_' + str(level2_count -1) + '_d3_' + str(level3_count -1) + '.id, frequency: 0.1) \n'
				fout.write(string)

		linecount = linecount + 1

			
	except:
		raise