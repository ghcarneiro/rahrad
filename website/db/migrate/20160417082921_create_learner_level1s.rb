class CreateLearnerLevel1s < ActiveRecord::Migration
  def change
    create_table :learner_level1s do |t|
      t.string :name 
      t.references :dx_level1, index: true, foreign_key: true
      t.references :user, index: true, foreign_key: true
      t.integer :cases_attempted
      t.integer :correct_dx
      t.integer :excellent_cases
      t.timestamps null: false
    end
  end
end
