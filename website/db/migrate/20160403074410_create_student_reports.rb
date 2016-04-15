class CreateStudentReports < ActiveRecord::Migration
  def change
    create_table :student_reports do |t|
      t.string :report_text
      t.boolean :diagnosis_found
      t.integer :correct_sentences #Serialized array
      t.integer :missing_sentences #Serialized array
      t.references :expert_report, index: true, foreign_key: true
      t.references :learner_dx, index: true, foreign_key: true
      t.references :user, index: true, foreign_key: true
      t.timestamps null: false
    end
  end
end
