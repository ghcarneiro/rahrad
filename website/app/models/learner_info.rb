class LearnerInfo < ActiveRecord::Base
  has_and_belongs_to_many :expert_reports
  has_one :user
  serialize :report_array, Array
  serialize :answers_array, Array
end
