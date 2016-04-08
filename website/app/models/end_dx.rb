class EndDx < ActiveRecord::Base
	belongs_to :dxable, polymorphic: true
	has_many :learner_dxes


end
