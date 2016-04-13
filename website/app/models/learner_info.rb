class LearnerInfo < ActiveRecord::Base
  belongs_to :expert_report
  has_one :user
end
