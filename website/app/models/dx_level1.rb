class DxLevel1 < ActiveRecord::Base
  	has_many :learner_level1s
	has_many :dx_level2s
	has_many :end_dxes, :as => :dxable

	def self.search(search)
		where("name LIKE ?", "%#{search}%")
	end
end
