class DxLevel1 < ActiveRecord::Base
	has_many :dx_level2s
	has_many :end_dxes
end
