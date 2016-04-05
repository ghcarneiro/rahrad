class EndDx < ActiveRecord::Base
	belongs_to :dxable, polymorphic: true
	has_many :learner_dxes

	def self.search(search)
		where("name LIKE ?", "%#{search}%")
	end
end
