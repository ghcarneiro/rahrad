class LearnerInfo < ActiveRecord::Base
  has_and_belongs_to_many :expert_reports
  has_one :user
end
