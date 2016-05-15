class User < ActiveRecord::Base
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :trackable, :validatable

  validates :year_of_training, :presence => true,
                               :on => :create
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

  after_create :create_learner_info
  after_create :create_learner_level1s

  private

  def create_learner_level1s
    @dxlist = DxLevel1.all
    @dxlist.each do |dx|
      LearnerLevel1.create(:user_id => self.id, :dx_level1_id => dx.id, :name => dx.name, :cases_attempted => 0, :correct_dx => 0, :excellent_cases => 0)
    end
  end

  def create_learner_info
    LearnerInfo.create(:user_id => self.id, :test => true)
  end

end
