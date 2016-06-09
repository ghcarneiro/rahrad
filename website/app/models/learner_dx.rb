class LearnerDx < ActiveRecord::Base
	belongs_to :end_dx
	belongs_to :user
        has_many :student_reports
	cattr_accessor :current_user

	scope :spaced, -> {where("updated_at < ?", Time.now-7.days)}
	scope :week_ago, -> {where("updated_at < ?", Time.now-7.days)}
	scope :not_seen, -> {where("cases_attempted LIKE 0")}
	scope :needs_practice, -> {where("cases_attempted > 0 AND accuracy < 0.5 AND updated_at < ?", Time.now-3.days)}

	# Search for diagnoses by year level
	def self.dxable_search(year_level)
		if year_level == "1"  
			joins(:end_dx).where("end_dxes.category LIKE 'key'").order('end_dxes.category ASC')
		elsif year_level == "2"
			joins(:end_dx).where("end_dxes.category LIKE '1' OR category LIKE 'key'").order('end_dxes.category DESC')
			# Key conditions come after Cat 1 in ascending order
		elsif year_level == "3"
			joins(:end_dx).where("end_dxes.category LIKE '1' OR category LIKE '2'").order('end_dxes.category ASC')
		elsif year_level == "4"
			joins(:end_dx).where("end_dxes.category LIKE '2' OR category LIKE '3'").order('end_dxes.category ASC')
		elsif year_level == "5"
			joins(:end_dx).where("end_dxes.category LIKE '3'").order('end_dxes.category ASC')
		end
	end

end
