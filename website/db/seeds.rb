# This file should contain all the record creation needed to seed the database with its default values.
# The data can then be loaded with the rake db:seed (or created alongside the db with db:setup).
#
# Examples:
#
#   cities = City.create([{ name: 'Chicago' }, { name: 'Copenhagen' }])
#   Mayor.create(name: 'Emanuel', city: cities.first)


# Abdominal
d1 = DxLevel1.create(name: 'Abdominal', all_total: 75.4, all_good: 64.7, all_excellent: 45.5, all_number: 367, key_total: 95.4, key_good: 75.3, key_excellent: 70.2, key_number: 244, cat1_total: 65.4, cat1_good: 61.3, cat1_excellent: 58.0, cat1_number: 97, cat2_total: 61.0, cat2_good: 49.3, cat2_excellent: 20.1, cat2_number: 13, cat3_total: 38.1, cat3_good: 15.4, cat3_excellent: 5.1, cat3_number: 13)
EndDx.create(name: 'Free Gas', category: 'key', dx_level1_id: d1.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Bowel obstruction', category: 'key', dx_level1_id: d1.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Volvulus', category: 'key', dx_level1_id: d1.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Trauma to Abdominal Organs', category: 'key', dx_level1_id: d1.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'AAA and Rupture', category: 'key', dx_level1_id: d1.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)
EndDx.create(name: 'Mesenteric/Other Organ Ischaemia', category: 'key', dx_level1_id: d1.id, total: 76.4, good: 69.0, excellent: 55.7, number: 5)
EndDx.create(name: 'Testicular Torsion', category: 'key', dx_level1_id: d1.id, total: 43.9, good: 29.0, excellent: 10.5, number: 19)
EndDx.create(name: 'Abdominal Sepsis', category: 'key', dx_level1_id: d1.id, total: 68.4, good: 45.7, excellent: 30.5, number: 23)
EndDx.create(name: 'Hydronephrosis/Pyonephrosis', category: 'key', dx_level1_id: d1.id, total: 20.9, good: 15.4, excellent: 7.5, number: 8)
EndDx.create(name: 'Acute Abdomen', category: 'key', dx_level1_id: d1.id, total: 38.9, good: 28.5, excellent: 18.9, number: 12)

# Cardiothoracic
d2 = DxLevel1.create(name: 'Cardiothoracic', all_total: 95.4, all_good: 73.1, all_excellent: 59.0, all_number: 210, key_total: 98.4, key_good: 80.3, key_excellent: 75.5, key_number: 150, cat1_total: 85.0, cat1_good: 80.8, cat1_excellent: 70.4, cat1_number: 48, cat2_total: 69.0, cat2_good: 53.3, cat2_excellent: 40.1, cat2_number: 10, cat3_total: 21.0, cat3_good: 9.6, cat3_excellent: 2.4, cat3_number: 2)
EndDx.create(name: 'Pneumothorax', category: 'key', dx_level1_id: d2.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Pulmonary Embolus', category: 'key', dx_level1_id: d2.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Pulmonary Oedema', category: 'key', dx_level1_id: d2.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Aortic/Vascular Trauma', category: 'key', dx_level1_id: d2.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'Aortic Dissection', category: 'key', dx_level1_id: d2.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)
EndDx.create(name: 'Cardiac Tamponade', category: 'key', dx_level1_id: d2.id, total: 76.4, good: 69.0, excellent: 55.7, number: 5)
EndDx.create(name: 'Typical/Atypical Infections', category: 'key', dx_level1_id: d2.id, total: 43.9, good: 29.0, excellent: 10.5, number: 19)
EndDx.create(name: 'Flail Chest', category: 'key', dx_level1_id: d2.id, total: 68.4, good: 45.7, excellent: 30.5, number: 23)

# Head and Neck
d3 = DxLevel1.create(name: 'Extracranial Head and Neck', all_total: 70.1, all_good: 50.3, all_excellent: 31.2, all_number: 137, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Foreign Bodies', category: 'key', dx_level1_id: d3.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Retropharyngeal Abscesses', category: 'key', dx_level1_id: d3.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Tonsillar Abscess', category: 'key', dx_level1_id: d3.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Ruptured Oesophagus', category: 'key', dx_level1_id: d3.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)

# Neuroradiology
d4 = DxLevel1.create(name: 'Neuroradiology', all_total: 40.1, all_good: 20.3, all_excellent: 5.0, all_number: 25, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Subarachnoid Haemorrhage', category: 'key', dx_level1_id: d4.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Extradural/Subdural Haemorrhage', category: 'key', dx_level1_id: d4.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Cerebral Parenchymal Haemorrhage', category: 'key', dx_level1_id: d4.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Cerebral Abscess/Empyema', category: 'key', dx_level1_id: d4.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'Venous Sinus Thrombosis', category: 'key', dx_level1_id: d4.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)
EndDx.create(name: 'Herniation', category: 'key', dx_level1_id: d4.id, total: 76.4, good: 69.0, excellent: 55.7, number: 5)
EndDx.create(name: 'Stroke', category: 'key', dx_level1_id: d4.id, total: 43.9, good: 29.0, excellent: 10.5, number: 19)
EndDx.create(name: 'Diffuse Axonal Injury', category: 'key', dx_level1_id: d4.id, total: 68.4, good: 45.7, excellent: 30.5, number: 23)
EndDx.create(name: 'Hypoxic Brain Injury', category: 'key', dx_level1_id: d4.id, total: 20.9, good: 15.4, excellent: 7.5, number: 8)
EndDx.create(name: 'Cord Compression', category: 'key', dx_level1_id: d4.id, total: 38.9, good: 28.5, excellent: 18.9, number: 12)
EndDx.create(name: 'Cerebrovascular Injury', category: 'key', dx_level1_id: d4.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)

# Musculoskeletal
d5 = DxLevel1.create(name: 'Musculoskeletal', all_total: 65, all_good: 60, all_excellent: 59.8, all_number: 137, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Fracture/Dislocation', category: 'key', dx_level1_id: d5.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Spine Fracture/Dislocation', category: 'key', dx_level1_id: d5.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Spinal Cord Injuries', category: 'key', dx_level1_id: d5.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Epidural Haematomas', category: 'key', dx_level1_id: d5.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'Septic Arthritis/Osteomyelitis/Discitis', category: 'key', dx_level1_id: d5.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)

# Paediatric
d6 = DxLevel1.create(name: 'Paediatric', all_total: 81.3, all_good: 42.0, all_excellent: 35.0, all_number: 137, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Airway Foreign Body', category: 'key', dx_level1_id: d6.id, total: 86.4, good: 60.7, excellent: 40.5, number: 30)
EndDx.create(name: 'Intussusception', category: 'key', dx_level1_id: d6.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Pyloric Stenosis', category: 'key', dx_level1_id: d6.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Malrotation', category: 'key', dx_level1_id: d6.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Epiglotitis/Croup', category: 'key', dx_level1_id: d6.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'NEC', category: 'key', dx_level1_id: d6.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'Non-Accidental Injury', category: 'key', dx_level1_id: d6.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)

# Obstetrics and Gynaecology
d7 = DxLevel1.create(name: 'Obstetrics and Gynaecology', all_total: 65.5, all_good: 50.9, all_excellent: 45.6, all_number: 137, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Viability Scanning', category: 'key', dx_level1_id: d7.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Ectopic Pregnancy', category: 'key', dx_level1_id: d7.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Tubo-Ovarian Masses', category: 'key', dx_level1_id: d7.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Ovarian Masses', category: 'key', dx_level1_id: d7.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
EndDx.create(name: 'Ruptured Ovarian Cysts', category: 'key', dx_level1_id: d7.id, total: 96.4, good: 76.7, excellent: 65.5, number: 49)

# Vascular and Interventional Radiology
d8 = DxLevel1.create(name: 'Vascular and Interventional Radiology', all_total: 59.6, all_good: 47.5, all_excellent: 25.6, all_number: 137, key_total: 85.9, key_good: 65.1, key_excellent: 47.0, key_number: 59, cat1_total: 50.1, cat1_good: 40.9, cat1_excellent: 38.7, cat1_number: 40, cat2_total: 65.9, cat2_good: 41.8, cat2_excellent: 10.7, cat2_number: 30, cat3_total: 41.3, cat3_good: 15.9, cat3_excellent: 5.0, cat3_number: 8)
EndDx.create(name: 'Acute Aortic Syndromes', category: 'key', dx_level1_id: d8.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'Trauma', category: 'key', dx_level1_id: d8.id, total: 61.4, good: 59.7, excellent: 40.5, number: 24)
EndDx.create(name: 'GI Bleed', category: 'key', dx_level1_id: d8.id, total: 53.4, good: 21.7, excellent: 5.5, number: 13)
EndDx.create(name: 'Thromboembolism', category: 'key', dx_level1_id: d8.id, total: 30.8, good: 20.7, excellent: 12.4, number: 31)
