class LearnerDx < ActiveRecord::Base
	belongs_to :end_dx
	belongs_to :user
end
