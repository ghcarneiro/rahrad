import csv

FILENAME = "keydx-reports.csv"
count = 0
array = []

with open(FILENAME,'rb') as fin:
    try:
        for row in csv.reader(fin):
		# Extract levels
		if "Free gas" in row[2]:
			end_dx = "keydx_0"
		elif "Bowel obstruction" in row[2]:
			end_dx = "keydx_1"
		elif "Volvulus" in row[2]:
			end_dx = "keydx_2"
		elif "Organ trauma" in row[2]:
			end_dx = "keydx_3"
		elif "Abdominal aortic aneurysm and rupture" in row[2]:
			end_dx = "keydx_4"
		elif "Mesenteric ischaemia" in row[2]:
			end_dx = "keydx_5"
		elif "Other organ ischaemia/infarction" in row[2]:
			end_dx = "keydx_6"
		elif "Testicular torsion" in row[2]:
			end_dx = "keydx_7"
		elif "Abdominal sepsis" in row[2]:
			end_dx = "keydx_8"
		elif "Abscess, collections and drainage" in row[2]:
			end_dx = "keydx_9"
		elif "Hydronephrosis" in row[2]:
			end_dx = "keydx_10"
		elif "Pyonephrosis" in row[2]:
			end_dx = "keydx_11"
		elif "Acute abdomen" in row[2]:
			end_dx = "keydx_12"

		string = 'exr_' + str(count) + ' = ExpertReport.create(report_number: "' + row[0] + '", report_text: "' + row[1] + '", end_dx_id: ' + end_dx + '.id, times_attempted: 0, correct_diagnosis: 0, difficulty: 0) \n'
		array.append(string)
		count = count + 1

    except:
        raise

with open("test-seed-reports.txt", 'wb') as fout:
    for i in range(len(array)):
        fout.write(array[i])
