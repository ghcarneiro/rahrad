class DxLevel2 < ActiveRecord::Base
	belongs_to :dx_level1
	has_many :dx_level3s
	has_many :end_dxes, :as => :dxable

	def self.search(search)
		where("name LIKE ?", "%#{search}%")
	end
end
