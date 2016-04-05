class DxLevel3 < ActiveRecord::Base
	belongs_to :dx_level2
	has_many :end_dxes, :as => :dxable

	def self.search(search)
		where("name LIKE ?", "%#{search}%")
	end
end
