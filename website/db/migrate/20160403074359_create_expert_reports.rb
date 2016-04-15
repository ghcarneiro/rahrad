class CreateExpertReports < ActiveRecord::Migration
  def change
    create_table :expert_reports do |t|
      t.string :report_number
      t.string :report_text
      t.string :report_image
      t.references :end_dx, index: true, foreign_key: true
      t.references :learner_info, index: true, foreign_key: true
      t.integer :times_attempted
      t.integer :correct_diagnosis
      t.decimal :difficulty
      t.timestamps null: false
    end
  end
end
