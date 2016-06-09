class AddSpacedDxToLearnerInfos < ActiveRecord::Migration
  def change
    add_column :learner_infos, :spaced_dx, :string
  end
end
