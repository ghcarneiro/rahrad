class AddReportArrayToLearnerInfos < ActiveRecord::Migration
  def change
    add_column :learner_infos, :report_array, :text, array: true, default: []
  end
end
