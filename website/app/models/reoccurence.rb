class Reoccurence < ActiveRecord::Base
	validates :query, :interval, :unit, :time, presence: true
	belongs_to :user
end