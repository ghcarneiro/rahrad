class LearnerDx < ActiveRecord::Base
	belongs_to :end_dx
	belongs_to :user
        has_many :student_reports
	cattr_accessor :current_user

	scope :spaced, -> {where("updated_at < ?", Time.now-7.days)}
	scope :week_ago, -> {where("updated_at < ?", Time.now-7.days)}
	scope :not_seen, -> {where("cases_attempted LIKE 0")}
	scope :needs_practice, -> {where("cases_attempted > 0 AND accuracy < 0.5 AND updated_at < ?", Time.now-3.days)}

end
