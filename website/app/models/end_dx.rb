class EndDx < ActiveRecord::Base
	belongs_to :dxable, polymorphic: true
	has_many :learner_dxes
        has_many :expert_reports

	# Search for diagnoses based on parent level (eg. concept hierarchy, dropdown lists)
	def self.dxable_search3(search_id, search_type, year_level)
		if year_level == "1"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '1' OR category LIKE 'key')", search_id, search_type)
		elsif year_level == "3"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '1' OR category LIKE '2')", search_id, search_type)
		end
	end

	# Search for diagnoses by name
	def self.dxable_search2(search, year_level)
		if year_level == "1"
			where("name LIKE ? AND (category LIKE '1' OR category LIKE 'key')", "%#{search}%").order('category DESC')
			# Key conditions come after Cat 1 in ascending order
		elsif year_level == "3"
			where("name LIKE ? AND (category LIKE '1' OR category LIKE '2')", "%#{search}%").order('category ASC')
		end
	end
end
