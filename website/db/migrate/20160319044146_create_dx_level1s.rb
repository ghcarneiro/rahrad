class CreateDxLevel1s < ActiveRecord::Migration
  def change
    create_table :dx_level1s do |t|
      t.string :name  # eg. Abdominal

      # Total for all diagnoses
      t.float :all_total # eg. 84.7% completed
      t.float :all_good # eg. 65.7% of reports with correct diagnosis
      t.float :all_excellent # eg. 50.6% of reports excellent
      t.integer :all_number # eg. 587 reports completed

      # Key diagnoses
      t.float :key_total # eg. 84.7% completed
      t.float :key_good # eg. 65.7% of reports with correct diagnosis
      t.float :key_excellent # eg. 50.6% of reports excellent
      t.integer :key_number # eg. 587 reports completed

      # Category 1 diagnoses
      t.float :cat1_total # eg. 84.7% completed
      t.float :cat1_good # eg. 65.7% of reports with correct diagnosis
      t.float :cat1_excellent # eg. 50.6% of reports excellent
      t.integer :cat1_number # eg. 587 reports completed

      # Category 2 diagnoses
      t.float :cat2_total # eg. 84.7% completed
      t.float :cat2_good # eg. 65.7% of reports with correct diagnosis
      t.float :cat2_excellent # eg. 50.6% of reports excellent
      t.integer :cat2_number # eg. 587 reports completed

      # Category 3 diagnoses
      t.float :cat3_total # eg. 84.7% completed
      t.float :cat3_good # eg. 65.7% of reports with correct diagnosis
      t.float :cat3_excellent # eg. 50.6% of reports excellent
      t.integer :cat3_number # eg. 587 reports completed

      t.timestamps null: false
    end
  end
end
