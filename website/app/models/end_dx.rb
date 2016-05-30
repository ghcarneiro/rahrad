class EndDx < ActiveRecord::Base
	belongs_to :dxable, polymorphic: true
	has_many :learner_dxes
        has_many :expert_reports

	# Search for diagnoses based on parent level (eg. concept hierarchy, dropdown lists)
	def self.dxable_search3(search_id, search_type, year_level)
		if year_level == "1"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE 'key')", search_id, search_type)
		elsif year_level == "2"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE 'key' OR category LIKE '1')", search_id, search_type)
		elsif year_level == "3"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '1' OR category LIKE '2')", search_id, search_type)
		elsif year_level == "4"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '2' OR category LIKE '3')", search_id, search_type)
		elsif year_level == "5"
			where("dxable_id LIKE ? AND dxable_type LIKE ? AND (category LIKE '3')", search_id, search_type)
		end
	end

	# For skill meters: Search for relevant diagnoses at any level of the hierarchy
	def self.dxable_search2a(l1_name, year_level)
		if year_level == "1"
			where("l1_name LIKE ? AND (category LIKE 'key')", l1_name)
		elsif year_level == "2"
			where("l1_name LIKE ? AND (category LIKE 'key' OR category LIKE '1')", l1_name)
		elsif year_level == "3"
			where("l1_name LIKE ? AND (category LIKE '1' OR category LIKE '2')", l1_name)
		elsif year_level == "4"
			where("l1_name LIKE ? AND (category LIKE '2' OR category LIKE '3')", l1_name)
		elsif year_level == "5"
			where("l1_name LIKE ? AND (category LIKE '3')", l1_name)
		end
	end

	def self.dxable_search2b(l2_name, year_level)
		if year_level == "1"
			where("l2_name LIKE ? AND (category LIKE 'key')", l2_name)
		elsif year_level == "2"
			where("l2_name LIKE ? AND (category LIKE 'key' OR category LIKE '1')", l2_name)
		elsif year_level == "3"
			where("l2_name LIKE ? AND (category LIKE '1' OR category LIKE '2')", l2_name)
		elsif year_level == "4"
			where("l2_name LIKE ? AND (category LIKE '2' OR category LIKE '3')", l2_name)
		elsif year_level == "5"
			where("l2_name LIKE ? AND (category LIKE '3')", l2_name)
		end
	end

	# There are some level 3 topics with the same name!!!
	def self.dxable_search2c(l2_name, l3_name, year_level)
		if year_level == "1"
			where("l2_name LIKE ? AND l3_name LIKE ? AND (category LIKE 'key')", l2_name, l3_name)
		elsif year_level == "2"
			where("l2_name LIKE ? AND l3_name LIKE ? AND (category LIKE 'key' OR category LIKE '1')", l2_name, l3_name)
		elsif year_level == "3"
			where("l2_name LIKE ? AND l3_name LIKE ? AND (category LIKE '1' OR category LIKE '2')", l2_name, l3_name)
		elsif year_level == "4"
			where("l2_name LIKE ? AND l3_name LIKE ? AND (category LIKE '2' OR category LIKE '3')", l2_name, l3_name)
		elsif year_level == "5"
			where("l2_name LIKE ? AND l3_name LIKE ? AND (category LIKE '3')", l2_name, l3_name)
		end
	end

	# Search for diagnoses by name
	def self.dxable_search2(search, year_level)
		if year_level == "1"  
			where("name LIKE ? AND (category LIKE 'key')", "%#{search}%").order('category ASC')
		elsif year_level == "2"
			where("name LIKE ? AND (category LIKE '1' OR category LIKE 'key')", "%#{search}%").order('category DESC')
			# Key conditions come after Cat 1 in ascending order
		elsif year_level == "3"
			where("name LIKE ? AND (category LIKE '1' OR category LIKE '2')", "%#{search}%").order('category ASC')
		elsif year_level == "4"
			where("name LIKE ? AND (category LIKE '2' OR category LIKE '3')", "%#{search}%").order('category ASC')
		elsif year_level == "5"
			where("name LIKE ? AND (category LIKE '3')", "%#{search}%").order('category ASC')
		end
	end
end
