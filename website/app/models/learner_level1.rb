class LearnerLevel1 < ActiveRecord::Base
	belongs_to :user
	belongs_to :dx_level1
end
