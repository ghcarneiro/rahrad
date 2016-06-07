class AddCurrentIndexToLearnerInfos < ActiveRecord::Migration
  def change
    add_column :learner_infos, :current_index, :integer
  end
end
