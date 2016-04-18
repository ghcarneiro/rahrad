level1 = "0"
count = 0

expert_no = 120000

with open("keydiagnosis-list","rb") as fin, open("test-seed-key.txt", "wb") as fout:
    for line in fin:
        try:
		# Extract levels
		if "LEVEL 1: ABDOMINAL" in line:
			level1 = "d1_0"
			levelname = "Abdominal"
		elif "LEVEL 1: CARDIOTHORACIC" in line:
			level1 = "d1_1"
			levelname = "Cardiothoracic"
		elif "LEVEL 1: EXTRACRANIAL HEAD & NECK" in line:
			level1 = "d1_2"
			levelname = "Extracranial head & neck"
		elif "LEVEL 1: NEURORADIOLOGY" in line:
			level1 = "d1_3"
			levelname = "Neuroradiology"
		elif "LEVEL 1: MUSCULOSKELETAL" in line:
			level1 = "d1_4"
			levelname = "Musculoskeletal"
		elif "LEVEL 1: PAEDIATRIC" in line:
			level1 = "d1_5"
			levelname = "Paediatric"
		elif "LEVEL 1: OBSTETRIC & GYNAECOLOGICAL" in line:
			level1 = "d1_7"
			levelname = "Obstetric & gynaecological"
		elif "LEVEL 1: VASCULAR" in line:
			level1 = "d1_8"
			levelname = "Vascular"

		else:
			line = line.strip()
			if (line != "") and (level1 != "0"):
				diagnosis = line.decode('utf-8', 'ignore').encode('utf-8') 
				string = 'keydx_' + str(count) + ' = EndDx.create(name: "' + diagnosis + '", category: "key", dxable_id: ' + level1 + '.id, l1_name: "' + levelname + '", l2_name: "0", l3_name: "0", dxable_type: "DxLevel1", frequency: 0.1) \n'
				fout.write(string)
				for x in range(0, 5):
					string2 = 'ExpertReport.create(report_number: "' + str(expert_no) + '", report_text: "Report no ' + str(expert_no) + '. This is a fake report about ' + diagnosis + '. This is a fake report. This is a fake report. Repeat, this is a fake report.", end_dx_id: keydx_' + str(count) + '.id, times_attempted: 0, correct_diagnosis: 0, difficulty: 0) \n'
					fout.write(string2)
					expert_no = expert_no + 1
				count = count + 1
	except:
		raise
