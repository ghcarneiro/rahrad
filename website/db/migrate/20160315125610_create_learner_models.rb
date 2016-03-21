class CreateLearnerModels < ActiveRecord::Migration
  def change
    create_table :learner_models do |t|
      t.integer :report_id
      t.string :level1 # Body systems: Abdominal, Cardiothoracic, etc (system level)
      t.string :level2 # Body systems -> Abdominal: Gastrointestinal, hepatopancreatobiliary, etc (multi-organ level)
      t.string :level3 # Body systems -> Abdominal -> Gastrointestinal: Oesophagus, Stomach, etc (organ level)
      t.string :level4 # eg. Body systems -> Abdominal -> Gastrointestinal -> Oesophagus: Carcinoma, Trauma, etc (diagnosis level)
      t.integer :correct # Diagnosis correct? 0 = no, 1 = yes
      t.float :accuracy # How many sentences were correct (percentage)
      t.datetime :submit_time
      t.text :learner_report
    end
  end
end


# For each diagnosis: 
# % correct: correct_total/5
# % incorrect: total_reports - correct
# % good: (number with accuracy > 90 and correct)/5
# % improvement: (number with accuracy < 90 && correct)/5

# To display on kiviat graph incorrect = incorrect + improvement + good
# improvement = improvement + good 
# Will then need to subtract the extra categories for accurate tooltips

# For each organ:
# good = good_total/number_of_subcategories
