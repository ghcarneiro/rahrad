class User < ActiveRecord::Base
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :trackable, :validatable

  # When the user object is destroyed, all its reoccurences will be destroy too
  has_many :reoccurences, dependent: :destroy

  # When the user object is destroyed, all its saved_searches will be destroy too
  has_many :saved_searches, dependent: :destroy

  has_many :learner_dxes, dependent: :destroy
  has_one :learner_info

  after_create :create_learner_dx
  after_create :create_learner_info

  private

  # Method for creating learner model when user is created
  def create_learner_dx
    @dxlist = EndDx.all
    @dxlist.each do |dx|
      LearnerDx.create(:user_id => self.id, :end_dx_id => dx.id, :name => dx.name, :review_list => false, :cases_attempted => 10, :correct_dx => 8, :excellent_cases => 3)
    end
  end

  def create_learner_info
    LearnerInfo.create(:user_id => self.id)
  end

end
