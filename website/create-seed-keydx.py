level1 = "0"
count = 0

with open("keydiagnosis-list","rb") as fin, open("test-seed-key.txt", "wb") as fout:
    for line in fin:
        try:
		# Extract levels
		if "LEVEL 1: ABDOMINAL" in line:
			level1 = "d1_0"
		elif "LEVEL 1: CARDIOTHORACIC" in line:
			level1 = "d1_1"
		elif "LEVEL 1: EXTRACRANIAL HEAD & NECK" in line:
			level1 = "d1_2"
		elif "LEVEL 1: NEURORADIOLOGY" in line:
			level1 = "d1_3"
		elif "LEVEL 1: MUSCULOSKELETAL" in line:
			level1 = "d1_4"
		elif "LEVEL 1: PAEDIATRIC" in line:
			level1 = "d1_5"
		elif "LEVEL 1: OBSTETRIC & GYNAECOLOGICAL" in line:
			level1 = "d1_7"
		elif "LEVEL 1: VASCULAR" in line:
			level1 = "d1_8"

		else:
			line = line.strip()
			if (line != "") and (level1 != "0"):
				diagnosis = line.decode('utf-8', 'ignore').encode('utf-8') 
				string = 'keydx_' + str(count) + ' = EndDx.create(name: "' + diagnosis + '", category: "key", dxable_id: ' + level1 + '.id, dxable_type: "DxLevel1", frequency: 0.1) \n'
				fout.write(string)
				count = count + 1
	except:
		raise
