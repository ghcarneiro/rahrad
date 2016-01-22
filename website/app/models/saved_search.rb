class SavedSearch < ActiveRecord::Base
	validates :query, :time, presence: true
	belongs_to :user
end