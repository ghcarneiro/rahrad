class ExpertReport < ActiveRecord::Base
    belongs_to :end_dx
    has_many :learner_infos
end
