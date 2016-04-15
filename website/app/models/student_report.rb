class StudentReport < ActiveRecord::Base
  belongs_to :expert_report
  belongs_to :learner_dx
  belongs_to :user
  serialize :correct_sentences, Array
  serialize :missing_sentences, Array
end
