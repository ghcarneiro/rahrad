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
  #after_create :create_learner_level1s

  private

  def create_learner_info
    @dxlist = DxLevel1.all
    @dxlist.each do |dx|
      LearnerLevel1.create(:user_id => self.id, :dx_level1_id => dx.id, :name => dx.name, :cases_attempted => 0, :correct_dx => 0, :excellent_cases => 0)
    end


    @learnerinfo = LearnerInfo.create(:user_id => self.id, :test => true)

    # TEST REPORTS
    @expert_renal = ExpertReport.create(report_number: "100000000", report_text: "THERE IS A HYPODENSE LESION IN SEGMENT 4A OF THE LIVER MEASURING 1.5CM IN DIAMETER.  THE APPEARANCES ARE CONSISTENT WITH THAT OF A SIMPLE CYST.  THE REMAINDER OF THE LIVER SHOWS NO EVIDENCE OF ANY ABNORMALITY. THERE IS A VERY LARGE MASS ON THE SUPERIOR PART OF THE LEFT KIDNEY MEASURING 8X9.5CM. THERE IS PATCHY ENHANCEMENT WITH CONTRAST, SUGGESTING SOME NECROSIS WITHIN THE CENTRE OF THE TUMOUR. APPEARANCES WOULD BE CONSISTENT WITH A RENAL CELL CARCINOMA. THERE IS GROSS ENLARGEMENT OF THE LEFT RENAL VEIN, SUGGESTING THAT THIS IS INVOLVED IN THE TUMOUROUS PROCESS. THE LEFT GONADAL VEIN HOWEVER IS ENHANCING WELL.  THERE IS NO EVIDENCE OF ANY OBSTRUCTION OF THE INFERIOR VENA CAVA. THERE IS NO EVIDENCE OF ANY ILLUSION WITHIN THE RIGHT KIDNEY.  THE SPLEEN, PANCREAS AND ADRENALS APPEAR NORMAL. THE AORTA IS NOT DILATED THROUGHOUT ITS LENGTH. THE PATIENT APPEARS TO HAVE HAD A HYSTERECTOMY IN THE PAST.  THERE IS NO EVIDENCE OF ANY OTHER ABNORMALITY, OR LYMPHADENOPATHY WITHIN THE ABDOMEN OR PELVIS.  COMMENT:  VERY LARGE MASS ON THE LEFT KIDNEY INVOLVING THE LEFT RENAL VEIN, CONSISTENT WITH A RENAL CELL CARCINOMA.", end_dx_id: 145, times_attempted: 0, correct_dx: 0, incorrect_dx: 0, difficulty: 0) 

					@ldx = LearnerDx.new
					@ldx.name = @expert_renal.end_dx.name
					@ldx.end_dx_id = 145
					@ldx.user_id = self.id
					@ldx.cases_attempted = 12
					@ldx.missed_dx = 8
					@ldx.accuracy = 33.33
					@ldx.correct_dx = 4
					@ldx.excellent_cases = 0
					@ldx.save

@learner_1 = LearnerLevel1.where(:dx_level1_id => 1).where(:user_id => self.id).first
@learner_1.correct_dx = @learner_1.correct_dx + 4
@learner_1.cases_attempted = 18
@learner_1.save
      LearnerLevel2.create(:user_id => self.id, :dx_level2_id => 3, :name => "Renal and urinary tract", :cases_attempted => 18, :correct_dx => 4, :excellent_cases => 0)
      LearnerLevel3.create(:user_id => self.id, :dx_level3_id => 12, :name => "Renal neoplasia", :cases_attempted => 12, :correct_dx => 4, :excellent_cases => 0)

# 4 correct renal reports
4.times do
StudentReport.create(:learner_dx_id => @ldx.id, :user_id => self.id, :expert_report_id => @expert_renal.id, :report_text => "THERE IS A VERY LARGE MASS ON THE SUPERIOR PART OF THE LEFT KIDNEY MEASURING 8X9.5CM. THERE IS PATCHY ENHANCEMENT WITH CONTRAST, SUGGESTING SOME NECROSIS WITHIN THE CENTRE OF THE TUMOUR. APPEARANCES WOULD BE CONSISTENT WITH A RENAL CELL CARCINOMA. THERE IS GROSS ENLARGEMENT OF THE LEFT RENAL VEIN, SUGGESTING THAT THIS IS INVOLVED IN THE TUMOUROUS PROCESS. THERE IS MILD DILATATION OF THE ABDOMINAL AORTA AT THE LEVEL OF L3. THE SPLEEN, PANCREAS AND ADRENALS APPEAR NORMAL. THE PATIENT APPEARS TO HAVE HAD A HYSTERECTOMY IN THE PAST.  THERE IS NO EVIDENCE OF ANY OTHER ABNORMALITY, OR LYMPHADENOPATHY WITHIN THE ABDOMEN OR PELVIS.  COMMENT:  VERY LARGE MASS ON THE LEFT KIDNEY INVOLVING THE LEFT RENAL VEIN, CONSISTENT WITH A RENAL CELL CARCINOMA.", :diagnosis_found => true, :correct_sentences => ["0", "1", "2", "3", "5", "6", "7", "8"], :missing_sentences => ["0", "1", "2", "7", "8", "9", "11"], :ignore_sentences => [], :score => 53.33)
end

# 8 incorrect renal reports
8.times do
StudentReport.create(:learner_dx_id => @ldx.id, :user_id => self.id, :expert_report_id => @expert_renal.id, :report_text => "THERE IS A HYPODENSE LESION IN SEGMENT 4A OF THE LIVER MEASURING 1.5CM IN DIAMETER.  THE APPEARANCES ARE CONSISTENT WITH THAT OF A SIMPLE CYST. THERE IS A VERY LARGE MASS ON THE SUPERIOR PART OF THE LEFT KIDNEY MEASURING 8X9.5CM. THE APPEARANCE IS CONSISTENT WITH A BENIGN CYST. THE ABDOMINAL AORTA IS DILATED AT THE LEVEL OF THE LEFT RENAL ARTERY. THE SPLEEN, PANCREAS AND ADRENALS APPEAR NORMAL. THE PATIENT APPEARS TO HAVE HAD A HYSTERECTOMY IN THE PAST.  THERE IS NO EVIDENCE OF ANY OTHER ABNORMALITY, OR LYMPHADENOPATHY WITHIN THE ABDOMEN OR PELVIS.  COMMENT:  VERY LARGE MASS ON THE LEFT KIDNEY CONSISTENT WITH BENIGN CYST.", :diagnosis_found => false, :correct_sentences => ["0", "1", "2", "5", "6", "7"], :missing_sentences => ["2", "4", "5", "6", "7", "8", "9", "11", "14"], :ignore_sentences => [], :score => 40.00)
end


@expert_renal2 = ExpertReport.create(report_number: "100000001", report_text: "SMALL CALCULI ARE SEEN WITHIN BOTH KIDNEYS BILATERALLY.  THERE IS NO HYDRONEPHROSIS.  NO SIGNIFICANT PERINEPHRIC OR PERIURETERIC CHANGES.  NO CONVINCING URETERIC CALCULI.  WITHIN THE PELVIS THE CALCIFIED DENSITIES ARE MORE IN KEEPING WITH PHLEBOLITHS.  THERE IS NO FREE INTRA-PERITONEAL FLUID. NO SIGNIFICANT DIVERTICULAR DISEASE.  THE ADRENAL GLANDS ARE NORMAL.  COMMENT:  SEVERAL RENAL CALCULI.  NO URETERIC STONE.", end_dx_id: 164, times_attempted: 0, correct_dx: 0, incorrect_dx: 0, difficulty: 0) 

					@ldx2 = LearnerDx.new
					@ldx2.name = @expert_renal2.end_dx.name
					@ldx2.end_dx_id = 164
					@ldx2.user_id = self.id
					@ldx2.cases_attempted = 6
					@ldx2.missed_dx = 6
					@ldx2.accuracy = 0
					@ldx2.correct_dx = 0
					@ldx2.excellent_cases = 0
					@ldx2.save


      LearnerLevel3.create(:user_id => self.id, :dx_level3_id => 15, :name => "Miscellaneous renal conditions", :cases_attempted => 6, :correct_dx => 0, :excellent_cases => 0)

# Incorrect renal calculus report
6.times do
StudentReport.create(:learner_dx_id => @ldx2.id, :user_id => self.id, :expert_report_id => @expert_renal2.id, :report_text => "SMALL CALCULI ARE SEEN WITHIN BOTH KIDNEYS BILATERALLY.  THERE IS NO HYDRONEPHROSIS.  NO SIGNIFICANT PERINEPHRIC OR PERIURETERIC CHANGES. THERE APPEARS TO BE A SMALL 1MM CALCULUS AT THE LEVEL OF THE RIGHT VUJ. THERE IS NO FREE INTRA-PERITONEAL FLUID. NO SIGNIFICANT DIVERTICULAR DISEASE.  THE ADRENAL GLANDS ARE NORMAL.  COMMENT:  SEVERAL RENAL CALCULI. 1MM RIGHT VUJ CALCULUS.", :diagnosis_found => false, :correct_sentences => ["0", "1", "2", "4", "5", "6", "7"], :missing_sentences => ["3", "4", "9"], :ignore_sentences => [], :score => 70.00)
end

@expert_renal3 = ExpertReport.create(report_number: "100000002", report_text: "A CRECENTRIC PERINEPHRIC COLLECTION IS SEEN SURROUNDING THE LEFT KIDNEY CONSISTENT WITH HAEMORRHAGE. THE HAEMORRHAGE IS CONTAINED WITHIN GEROTA'S FASCIA AND THERE IS NO FREE INTRAPERITONEAL FLUID. THE HAEMATOMA MEASURES APPROXIMATELY 1CM IN MAXIMAL WIDTH. NO OTHER INTRA ABDOMINAL ABNORMALITY SEEN.  COMMENT: LEFT PERINEPHRIC HAEMATOMA.", end_dx_id: 2023, times_attempted: 0, correct_dx: 0, incorrect_dx: 0, difficulty: 0) 

					@ldx3 = LearnerDx.new
					@ldx3.name = @expert_renal3.end_dx.name
					@ldx3.end_dx_id = 2023
					@ldx3.user_id = self.id
					@ldx3.cases_attempted = 3
					@ldx3.missed_dx = 3
					@ldx3.accuracy = 0
					@ldx3.correct_dx = 0
					@ldx3.excellent_cases = 0
					@ldx3.save
@learner_3 = LearnerLevel1.where(:dx_level1_id => 9).where(:user_id => self.id).first
@learner_3.correct_dx = 0
@learner_3.cases_attempted = 3
@learner_3.save
      LearnerLevel2.create(:user_id => self.id, :dx_level2_id => 70, :name => "Abdominal vascular", :cases_attempted => 3, :correct_dx => 0, :excellent_cases => 0)
# INCORRECT 3
3.times do
StudentReport.create(:learner_dx_id => @ldx3.id, :user_id => self.id, :expert_report_id => @expert_renal3.id, :report_text => "THE LIVER, SPLEEN AND OTHER ABDOMINAL ORGANS APPEAR UNREMARKABLE. NO EVIDENCE OF RENAL CALCULI.", :diagnosis_found => false, :correct_sentences => [], :missing_sentences => ["0", "1", "2", "3", "4"], :ignore_sentences => [], :score => 0.00)
end

  end

end
