class EndDx < ActiveRecord::Base
	belongs_to :dxable, polymorphic: true
	has_many :learner_dxes
        has_many :expert_reports

	# Year 1 content: key diagnoses and cat 1 diagnoses
	def self.dxable_search(search_id, search_type, year_level)
		if year_level == "1"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '1' OR category LIKE 'key')", search_id, search_type)
		elsif year_level == "3"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '1' OR category LIKE '2')", search_id, search_type)
		end
	end
end
