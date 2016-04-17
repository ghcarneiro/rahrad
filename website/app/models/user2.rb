class User < ActiveRecord::Base
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :trackable, :validatable

  # When the user object is destroyed, all its reoccurences will be destroy too
  has_many :reoccurences, dependent: :destroy

  # When the user object is destroyed, all its saved_searches will be destroy too
  has_many :saved_searches, dependent: :destroy

  has_one :learner_info, dependent: :destroy
  has_many :student_reports, dependent: :destroy
  has_many :learner_level1s, dependent: :destroy
  has_many :learner_level2s, dependent: :destroy
  has_many :learner_level3s, dependent: :destroy
  has_many :learner_dxes, dependent: :destroy

  after_create :create_learner_dx
  after_create :create_learner_info

  private

  # Method for creating learner model when user is created
  def create_learner_dx
    @dxlist = EndDx.all
    @dxlist.each do |dx|
      LearnerDx.create(:user_id => self.id, :end_dx_id => dx.id, :name => dx.name, :review_list => false, :cases_attempted => 0, :correct_dx => 0, :excellent_cases => 0)
    end
  end

  def create_learner_info
    LearnerInfo.create(:user_id => self.id)
  end

end
