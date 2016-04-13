class CreateLearnerInfos < ActiveRecord::Migration
  def change
    create_table :learner_infos do |t|
      t.references :user, index: true, foreign_key: true #User's current report
      t.references :expert_report, index: true, foreign_key: true #User's current report
      t.timestamps null: false
    end
  end
end
