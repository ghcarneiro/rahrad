class LearnerDx < ActiveRecord::Base
	belongs_to :end_dx
	belongs_to :user
        has_many :student_reports
	cattr_accessor :current_user


end
