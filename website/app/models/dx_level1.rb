class DxLevel1 < ActiveRecord::Base
	has_many :dx_level2s

	def self.search(search)
		where("name LIKE ?", "%#{search}%")
	end
end
