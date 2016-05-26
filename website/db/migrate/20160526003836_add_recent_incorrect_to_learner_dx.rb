class AddRecentIncorrectToLearnerDx < ActiveRecord::Migration
  def change
    add_column :learner_dxes, :recent_incorrect, :string
  end
end
