class User < ActiveRecord::Base
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :trackable, :validatable

  # When the user object is destroyed, all its reoccurences will be destroy too
  has_many :reoccurences, dependent: :destroy

  # When the user object is destroyed, all its saved_searches will be destroy too
  has_many :saved_searches, dependent: :destroy
  
end
