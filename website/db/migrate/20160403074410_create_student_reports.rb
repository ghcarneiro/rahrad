class CreateStudentReports < ActiveRecord::Migration
  def change
    create_table :student_reports do |t|
      t.string :report_text
      t.boolean :diagnosis_found
      t.text :correct_sentences, array: true, default: []
      t.text :missing_sentences, array: true, default: []
      t.text :ignore_sentences, array: true, default: []
      t.float :score
      t.references :expert_report, index: true, foreign_key: true
      t.references :learner_dx, index: true, foreign_key: true
      t.references :user, index: true, foreign_key: true
      t.timestamps null: false
    end
  end
end
