class CreateLearnerDxes < ActiveRecord::Migration
  def change
    create_table :learner_dxes do |t|
      t.string :name 
      t.references :end_dx, index: true, foreign_key: true
      t.references :user, index: true, foreign_key: true
      t.boolean :review_list
      t.integer :cases_attempted
      t.integer :missed_dx
      t.float :accuracy
      t.integer :correct_dx
      t.integer :excellent_cases
      t.timestamps null: false
    end
  end
end
