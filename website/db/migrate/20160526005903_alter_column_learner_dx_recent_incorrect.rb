class AlterColumnLearnerDxRecentIncorrect < ActiveRecord::Migration
  def self.up
      change_column :learner_dxes, :recent_incorrect, :float
  end
  def self.down
      change_column :learner_dxes, :recent_incorrect, :string
  end
end
