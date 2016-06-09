class AddAnswersArrayToLearnerInfos < ActiveRecord::Migration
  def change
    add_column :learner_infos, :answers_array, :text, array: true, default: []
  end
end
