class ProblemsController < ApplicationController
    	def index
		@ldx = LearnerDx.where(:review_list => true)		
    	end

	def user_select
		@noparam = false
		@nofound = false

		# Process search term
		if params[:search]
			if params[:search] == ""
				@noparam = true
			else
				@searchdx = EndDx.where(["name LIKE ?", "%#{params[:search]}%"])	
			end
			if !@searchdx.present?
				@nofound = true
			end
		end
			
	end

	# Displays all diagnoses on the trainee's review list
	def review_list
		@ldx = LearnerDx.where(:review_list => true).where(:user => current_user.id)
	end

	# Add diagnoses to the review list
	def review_list_add
		@count = 0 
		@noparam = false
		@nofound = false
		
		# Process search term
		if params[:search]
			if params[:search] == ""
				@noparam = true
			else
				@already_reviewed = LearnerDx.where(["name LIKE ?", "%#{params[:search]}%"]).where(:review_list => true).where(:user => current_user.id)
				@searchdx = EndDx.where(["name LIKE ?", "%#{params[:search]}%"])	
			end
			if !@searchdx.present?
				@nofound = true
			end
		end

		# Add selected diagnoses to review list
		if params[:id]
			@string = EndDx.where(:id => params[:id])
			@string.each do |s|
				@ldx = LearnerDx.where(:end_dx_id => s.id).first
				# If learner dx doesn't exist for this dx, create it
				if @ldx.nil?
					@ldx = LearnerDx.new
					@ldx.name = s.name
					@ldx.end_dx_id = s.id
					@ldx.user_id = current_user.id
					@ldx.cases_attempted = 0
					@ldx.correct_dx = 0
					@ldx.excellent_cases = 0
				end
				@ldx.review_list = true
				@ldx.save
				@count = @count + 1
			end
		end

		@ldx = LearnerDx.where(:review_list => true)
	end

	# Removes selected diagnoses from the review list
	def review_list_remove
		@count = 0 # Counts the number of diagnoses selected to determine whether the singular or plural of diagnosis should be used
		@nodx = true # If no diagnoses have been selected for removal, no additional text will be displayed
		
		if params[:id]
			@nodx = false
			@string = LearnerDx.where(:id => params[:id], :user => current_user.id)
			@string.each do |s|
				s.review_list = false
				s.save
				@count = @count + 1
			end
		end

		@ldx = LearnerDx.where(:review_list => true)
	end

	# Displays cases for the trainee to write reports on and processes their answers
	def cases
		@learnerinfo = LearnerInfo.where(:user_id => current_user.id).first

		# Shows a new problem to the trainee based on the diagnoses they have selected
		if params[:id]
			# Remove old current report
			if @currentreport.present?
				@currentreport = "0"
			end

			# Disconnect old selected reports from the learner info
			@oldreports = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@oldreports.each do |r|
				r.learner_info_id = "-1"
				r.save
			end

			# Connect reports from the new selected diagnoses to the learner info
			@string = EndDx.where(:id => params[:id])
			@string.each do |n|
				@select = ExpertReport.where(:end_dx_id => n.id)
				@select.each do |s|
					s.learner_info_id = @learnerinfo.id
					s.save
				end
			end

			# Select a random report from the selected diagnoses and set as the new current report
			@ids = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@currentreport = ExpertReport.find(@ids.sample)
			@learnerinfo.expert_report_id = @currentreport.id
			@learnerinfo.save

		# Processing the submitted student report
		elsif params[:q].present?
			@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
			@user_report = params[:q]

			# Create new student report in database
			r = StudentReport.new
			r.report_text = @user_report
			r.expert_report_id = @currentreport.id
			r.user_id = current_user.id
			@learnerdx = LearnerDx.where(:end_dx_id => @currentreport.end_dx_id).where(:user_id => current_user.id).first
			r.learner_dx_id = @learnerdx.id
			
			# Send request to python server
			@resultTemp = HTTP.post("http://localhost:5000", :json => { :expert_report => @currentreport.report_text, :learner_report => @user_report}).to_s

			# The response from the python server is the positions of the correct and missing sentences in their respective reports in a single array.
			# The correct sentences are separated from the missing sentences by the value -100.
			@missingsent = false # Tracks the transition from correct to missing sentences
			@resultTemp = @resultTemp.split(',')
			@resultTemp.pop
			@resultTemp.shift
			@resultTemp.each do |t|
				if (t == '-100') or (t == '\"-100')
					@missingsent = true
				elsif @missingsent != true
					r.correct_sentences << t
				else
					r.missing_sentences << t
				end
			end

			@percentage = ((r.correct_sentences.length).to_f/(r.correct_sentences.length + r.missing_sentences.length)*100)
			r.score = @percentage
		
			# TEMPORARY FILL-IN CODE ONLY FOR TESTING PURPOSES - WILL BE REPLACED BY PROPER CLASSIFIER
			# Uses some randomness to decide whether the report diagnosis is correct/incorrect
			if @percentage > 70
				r.diagnosis_found = true
			elsif @percentage < 30
				r.diagnosis_found = false
			elsif (Random.rand(10) > 5)
				r.diagnosis_found = true
			else
				r.diagnosis_found = false
			end

			# Save student report
			r.save
			@studentreport = r

			# Check if Level 1, 2 and 3 exist for the diagnosis in the learner model - if not, create them
			@enddx = EndDx.where(:id => @learnerdx.end_dx_id).first
			if @enddx.category != "key"
				@dxlevel3 = DxLevel3.where(:id => @enddx.dxable_id).first
				if @dxlevel3.nil?
					@dxlevel3 = "NOEXIST"
					@dxlevel2 = DxLevel2.where(:id => @enddx.dxable_id).first
				else
				@dxlevel2 = DxLevel2.where(:id => @dxlevel3.dx_level2_id).first
				end
				@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
			else			
				@dxlevel1 = DxLevel1.where(:id => @enddx.dxable_id).first
				@dxlevel2 = "NOEXIST"
				@dxlevel3 = "NOEXIST"
			end

			@learner_l1 = LearnerLevel1.where(:dx_level1_id => @dxlevel1.id).where(:user_id => current_user.id).first

			if @dxlevel2 != "NOEXIST"
				@learner_l2 = LearnerLevel2.where(:dx_level2_id => @dxlevel2.id).where(:user_id => current_user.id).first
				if (@learner_l2.nil?) and (@enddx.category != "key")
					@learner_l2 = LearnerLevel2.new
					@learner_l2.name = @dxlevel2.name
					@learner_l2.dx_level2_id = @dxlevel2.id
					@learner_l2.user_id = current_user.id
					@learner_l2.cases_attempted = 0
					@learner_l2.correct_dx = 0
					@learner_l2.excellent_cases = 0
				end
			end

			if @dxlevel3 != "NOEXIST"
				@learner_l3 = LearnerLevel3.where(:dx_level3_id => @dxlevel3.id).where(:user_id => current_user.id).first
				if (@learner_l3.nil?) and (@enddx.category != "key")
					@learner_l3 = LearnerLevel3.new
					@learner_l3.name = @dxlevel3.name
					@learner_l3.dx_level3_id = @dxlevel3.id
					@learner_l3.user_id = current_user.id
					@learner_l3.cases_attempted = 0
					@learner_l3.correct_dx = 0
					@learner_l3.excellent_cases = 0
				end
			end

			# Increment learner model and expert report and save data
			@currentreport.times_attempted += 1
			@learnerdx.cases_attempted += 1
			@learner_l1.cases_attempted += 1
			if @learner_l2.present?
				@learner_l2.cases_attempted += 1
			end
			if @learner_l3.present?
				@learner_l3.cases_attempted += 1
			end

			if r.diagnosis_found == true
				@currentreport.correct_diagnosis += 1
				@learnerdx.correct_dx += 1
				@learner_l1.correct_dx += 1
				if @learner_l2.present?
					@learner_l2.correct_dx += 1
				end
				if @learner_l3.present?
					@learner_l3.correct_dx += 1
				end
				if @percentage > 70
					@learnerdx.excellent_cases += 1
					@learner_l1.excellent_cases += 1
					if @learner_l2.present?
						@learner_l2.excellent_cases += 1
					end
					if @learner_l3.present?
						@learner_l3.excellent_cases += 1
					end
				end
			end

			@currentreport.difficulty = @currentreport.correct_diagnosis/@currentreport.times_attempted.to_f
			@currentreport.save
			@learnerdx.save
			@learner_l1.save
			if @learner_l2.present?
				@learner_l2.save
			end
			if @learner_l3.present?
				@learner_l3.save
			end
			

			#Split text into sentences
			@expert_sentences = @currentreport.report_text.split(".")
			@student_sentences = r.report_text.split(".")
			
		else
			@ids = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@currentreport = ExpertReport.find(@ids.sample)
			@learnerinfo.expert_report_id = @currentreport.id
			@learnerinfo.save
		end
			
		@showdx = EndDx.where(:id => @currentreport.end_dx_id).first
	end
end
