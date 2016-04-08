class LearnerDx < ActiveRecord::Base
	belongs_to :end_dx
	belongs_to :user
	cattr_accessor :current_user


end
