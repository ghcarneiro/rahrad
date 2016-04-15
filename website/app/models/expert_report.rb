class ExpertReport < ActiveRecord::Base
    belongs_to :end_dx
    has_and_belongs_to_many :learner_infos
    has_many :student_reports
end
