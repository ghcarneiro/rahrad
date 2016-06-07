class ProblemsController < ApplicationController

	# Whenever learner diagnoses are searched for, the user_id MUST be
	# the current user's id - otherwise it will fetch data from other
	# users!!!!

	# You must check whether something is present before performing actions on it,
	# otherwise Rails will complain if it doesn't exist. eg. if !@next.blank?

	##################################################################################
	# Main problem selection page
	##################################################################################
    	def index
		@ldx = LearnerDx.where(:review_list => true, :user_id => current_user.id)		
    	end

	##################################################################################
	# Add to review list - response to AJAX request
	##################################################################################
	def add
	    @l_id = params[:l]
	    @add_dx = LearnerDx.where(:id => @l_id, :user_id => current_user.id).first
	    @add_dx.review_list = true
	    @add_dx.save
	    
	    @html = '<span class="key">Added to review list</span><br/><span class="btn btn-danger btn-sm"><span class="glyphicon glyphicon-minus"></span> Remove from review list</span>'
	    respond_to do |format|
  	    	format.html do
		    render text: @html 
		end
	    end
	end

	##################################################################################
	# Remove from review list - response to AJAX request
	##################################################################################
	def remove
	    @l_id = params[:l]
	    @add_dx = LearnerDx.where(:id => @l_id, :user_id => current_user.id).first
	    @add_dx.review_list = false
	    @add_dx.save
	    
	    @html = '<span class="error">Removed from review list</span><br/><span class="btn btn-success btn-sm"><span class="glyphicon glyphicon-plus"></span> Add to review list</span>'
	    respond_to do |format|
  	    	format.html do
		    render text: @html 
		end
	    end
	end

	##################################################################################
	# Retrieve data for dropdown menus - response to AJAX request
	##################################################################################
	def data

	    # The parameter sent contains the level of the topic clicked (level 1, 2 or 3) and the id for that topic
	    @s = params[:l].split("_")
	    @level = @s[0]
	    @id = @s[1]
	    params[:search_id] = @id.to_i
	    params[:year_level] = current_user.year_of_training
	    @isend = 0

	    # Search for the appropriate data
	    if @level == "l1"
		@next = DxLevel2.where(:dx_level1_id => @id.to_i)
		params[:search_type] = "DxLevel1"
		@keydx = EndDx.dxable_search3(params[:search_id], params[:search_type], params[:year_level])
	    elsif @level == "l2"
	    	@next = DxLevel3.where(:dx_level2_id => @id.to_i)
		if @next.blank?
			params[:search_type] = "DxLevel2"
			@next = EndDx.dxable_search3(params[:search_id], params[:search_type], params[:year_level])
		    	@isend = 1
		end
	    elsif @level == "l3"
		params[:search_type] = "DxLevel3"
	    	@next = EndDx.dxable_search3(params[:search_id], params[:search_type], params[:year_level])
	    end
	    @html= "<div class='subdata' style='margin-left: 50px'>"

	    if !@next.blank?
	    @next.each do |n|
		@html = @html + "<table width='100%'><tr><td>"

		# Create ids in the same manner as the initial param sent (level + id)
	    	if @level == "l1"
		    @i = "l2_" + n.id.to_s
		elsif @level == "l2"
		    if @isend == 1
		        @i = "e_" + n.id.to_s
		    else
		    	@i = "l3_" + n.id.to_s
		    end
		elsif @level == "l3"
		        @i = "e_" + n.id.to_s
		end
	    
		# Code for topics (that are not diagnoses)
		if !@i.include?("e_")
  	    	@html = @html + "<span class='dx-toggle' id='" + @i + "'>" + n.name + "<span class='glyphicon glyphicon-menu-right'></span></span></td></tr></table>"
		else
		    @ldx = LearnerDx.where(:end_dx_id => n.id, :user_id => current_user.id, :review_list => true)
		    if !@ldx.blank?
			@checkbox = " <span class='key'>Already in review list</span>"
		    else
			@checkbox = " <input id='" + n.id.to_s + "' name=id[] value='" + n.id.to_s + "' type='checkbox'></input>"
		    end
		    @html = @html + "<strong>Category " + n.category + "</strong><label for='" + n.name + "'>" + n.name + "</label>" + @checkbox + "</td></tr></table>"
		end
	    end
	    else
		@html = @html + "No diagnoses could be found."
	    end

	    # Code to display key diagnoses if appropriate
 	    if @level == "l1" and (current_user.year_of_training == "1" or current_user.year_of_training == "2")
	    	@keydx.each do |k|
		    @ldx = LearnerDx.where(:end_dx_id => k.id, :user_id => current_user.id, :review_list => true)
		    if !@ldx.blank?
			@checkbox = " <span class='key'>Already in review list</span>"
		    else
			@checkbox = " <input id='" + k.id.to_s + "' name=id[] value='" + k.id.to_s + "' type='checkbox'></input>"
		    end
		    @html = @html + "<table width='100%'><tr><td><strong class='key-strong'>Key condition</strong><label for='" + k.name + "'>" + k.name + "</label>" + @checkbox + "</td></tr></table>"
	    	end
	    end

		
	    # Send AJAX response
	    @html = @html + "</div>"
	    respond_to do |format|
  	    	format.html do
		    render text: @html 
		end
  	    	format.json do
   	    	    render :json => @next
  	    	end
	    end
end
	##################################################################################
	# User select page - now 'Search for cases' page
	##################################################################################
	def user_select
		@noparam = false
		@nofound = false

		if params[:dropdown]
	    		@dxlevel1s = DxLevel1.all
		end

		# Process search term
		if params[:search]
			params[:year_level] = current_user.year_of_training
			if params[:search] == ""
				@noparam = true
			else
				@searchdx = EndDx.dxable_search2(params[:search], params[:year_level])	
			end
			if !@searchdx.present?
				@nofound = true
			end
		end
			
	end

	##################################################################################
	# Displays all diagnoses on the trainee's review list
	##################################################################################
	def review_list
		@pagetype = "list"
		@ldx = LearnerDx.where(:review_list => true, :user => current_user.id)
	end

	##################################################################################
	# Add diagnoses to the review list (data for the actual page)
	##################################################################################
	def review_list_add
		@pagetype = "add"
		@count = 0 
		@noparam = false
		@nofound = false
		
		if params[:dropdown]
	    		@dxlevel1s = DxLevel1.all
		end
		
		# Process search term
		if params[:search]
			params[:year_level] = current_user.year_of_training
			if params[:search] == ""
				@noparam = true
			else
				@already_reviewed = LearnerDx.where(["name LIKE ?", "%#{params[:search]}%"]).where(:review_list => true).where(:user_id => current_user.id)
				@searchdx = EndDx.dxable_search2(params[:search], params[:year_level])	
			end
			if !@searchdx.present?
				@nofound = true
			end
		end

		# Add selected diagnoses to review list and save
		if params[:id]
			@string = EndDx.where(:id => params[:id])
			@string.each do |s|
				@ldx = LearnerDx.where(:end_dx_id => s.id, :user => current_user.id).first
				# If learner dx doesn't exist for this dx, create it
				if @ldx.nil?
					@ldx = LearnerDx.new
					@ldx.name = s.name
					@ldx.end_dx_id = s.id
					@ldx.user_id = current_user.id
					@ldx.cases_attempted = 0
					@ldx.missed_dx = 0
					@ldx.accuracy = 0
					@ldx.correct_dx = 0
					@ldx.excellent_cases = 0
					@ldx.recent_incorrect = 0
					@ldx.recent_correct = 0
					@ldx.recent_excellent = 0
				end
				@ldx.review_list = true
				@ldx.save
				@count = @count + 1
			end
		end

		@ldx = LearnerDx.where(:review_list => true, :user_id => current_user.id)
	end

	##################################################################################
	# Removes selected diagnoses from the review list (data for the actual page)
	##################################################################################
	def review_list_remove
		@pagetype = "remove"
		@count = 0 # Counts the number of diagnoses selected to determine whether the singular or plural of diagnosis should be used
		@nodx = true # If no diagnoses have been selected for removal, no additional text will be displayed
		
		if params[:id]
			@nodx = false
				@string = LearnerDx.where(:id => params[:id], :user_id => current_user.id)
			@string.each do |s|
				s.review_list = false
				s.save
				@count = @count + 1
			end
			if @count == 0
				@count = -1
			end
		end

		@ldx = LearnerDx.where(:review_list => true, :user_id => current_user.id)

	end

	##################################################################################
	# Displays cases for the trainee to write reports on and processes their answers
	# REPORT ANALYSIS PART HERE
	##################################################################################
	def cases
		@learnerinfo = LearnerInfo.where(:user_id => current_user.id).first

		# SYSTEM SELECT - SYSTEM DECIDES WHAT DIAGNOSES TO SHOW #
		if params[:system_select]
			@dx_array = Array.new
			@practice_array = Array.new

			# Clear old report array
			@learnerinfo.report_array.length.times do
				@learnerinfo.report_array.pop
			end

			# Get diagnoses
			@spaced = LearnerDx.spaced.where(:user_id => current_user.id).first
			@number = 0

			if @spaced.nil?
				@string = LearnerDx.needs_practice.where(:user_id => current_user.id).limit(5).order("RANDOM()")
				@number = 5
			else
				@string = LearnerDx.needs_practice.where(:user_id => current_user.id).limit(4).order("RANDOM()")
				@number = 4
			end

			# Add to array - round 1
			@string.each do |n|
				@end = EndDx.where(:id => n.end_dx_id).first
				@dx_array << @end.id
				@practice_array << @end.id
				@number -= 1
			end

			# Get other dx to fill array if needed
			@number.times do
					# Get dx not previously done before
					@end = EndDx.joins(:learner_dxes).where("learner_dxes.user_id != ?", current_user.id).sample
					if !@end.nil?
						@learnerdx = LearnerDx.new
						@learnerdx.name = @end.name
						@learnerdx.end_dx_id = @end.id
						@learnerdx.user_id = current_user.id
						@learnerdx.cases_attempted = 0
						@learnerdx.correct_dx = 0
						@learnerdx.missed_dx = 0
						@learnerdx.accuracy = 0
						@learnerdx.excellent_cases = 0
						@learnerdx.recent_incorrect = 0
						@learnerdx.recent_correct = 0
						@learnerdx.recent_excellent = 0
						@learnerdx.save
					else
						# Get any dx
						@end = EndDx.sample
					end
					@dx_array << @end.id
					@practice_array << @end.id
					@number -= 1
			end



			if @spaced.nil?
				5.times do
					@learnerinfo.report_array << @practice_array.shift
				end
			else
				3.times do
					@learnerinfo.report_array << @practice_array.shift
				end
				@end = EndDx.where(:id => @spaced.end_dx_id).first
				@dx_array << @end.id
				@learnerinfo.report_array << @end.id
				@learnerinfo.report_array << @practice_array.shift
			end


			# Add to array - round 2
			# Make sure fifth and sixth report are not the same
			@element = @dx_array[-1]
			until @element != @dx_array[-1] do
				@element = @dx_array.sample
			end

			# Add sixth report to array
			@dx_array.delete_at(@dx_array.index(@element))
			@learnerinfo.report_array << @element

			# Add last four reports to array
			4.times do
				@element = @dx_array.sample
				@dx_array.delete_at(@dx_array.index(@element))
				@learnerinfo.report_array << @element
			end

			@learnerinfo.current_index = 1
			@learnerinfo.save

		end

		# USER SELECT OR REVIEW LIST DIAGNOSES #
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

		# REPORT ANALYSIS SECTION #
		elsif params[:q].present?
			@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
			@user_report = params[:q]

			# Create new student report object in database
			r = StudentReport.new
			r.report_text = @user_report
			r.expert_report_id = @currentreport.id
			r.user_id = current_user.id
			@learnerdx = LearnerDx.where(:end_dx_id => @currentreport.end_dx_id).where(:user_id => current_user.id).first

			# Create learner dx if it does not exist yet
			if @learnerdx.nil?
					@e = EndDx.where(:id => @currentreport.end_dx_id).first
					@learnerdx = LearnerDx.new
					@learnerdx.name = @e.name
					@learnerdx.end_dx_id = @e.id
					@learnerdx.user_id = current_user.id
					@learnerdx.cases_attempted = 0
					@learnerdx.correct_dx = 0
					@learnerdx.missed_dx = 0
					@learnerdx.accuracy = 0
					@learnerdx.excellent_cases = 0
					@learnerdx.recent_incorrect = 0
					@learnerdx.recent_correct = 0
					@learnerdx.recent_excellent = 0
					@learnerdx.save
			end
			@learnerdx = LearnerDx.where(:end_dx_id => @currentreport.end_dx_id).where(:user_id => current_user.id).first
			r.learner_dx_id = @learnerdx.id

			
			# PROPER CODE FOR SENDING REPORT TO PYTHON SERVER FOR ANALYSIS - DOES NOT RUN IN TEST MODE #
			if current_user.learner_info.test == false

			# Send request to python server - response is stored in @resultTemp
			@resultTemp = HTTP.post("http://localhost:5000", :json => { :expert_report => @currentreport.report_text, :learner_report => @user_report}).to_s

			# The response from the python server is the positions of the correct and missing sentences in their respective reports in a single array.
			# The correct sentences are separated from the missing sentences by the value -100.

			# Tracks the transition from correct to missing sentences
			@missingsent = false 
			@resultTemp = @resultTemp.split(',')
			# Pop/shift gets rid of some unnecessary info from the array - I forgot what it is exactly
			@resultTemp.pop
			@resultTemp.shift

			# Put values into the appropriate array - correct or missing sentences
			@resultTemp.each do |t|
				if (t == '-100') or (t == '\"-100')
					@missingsent = true
				elsif @missingsent != true
					r.correct_sentences << t
				else
					r.missing_sentences << t
				end
			end

			#Split text into sentences
			@expert_sentences = @currentreport.report_text.split(".")
			@student_sentences = r.report_text.split(".")
			@correct_sentences = @expert_sentences.length - r.missing_sentences.length

			# Calculate score for overall report correctness
			@percentage = (@correct_sentences.to_f/@expert_sentences.length)*100
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

			###################################################################################
			# FOR TEST CODE ONLY - MOCK DATA   (used for usability testing)
			###################################################################################
			else
				if current_user.learner_info.test == true and (@currentreport.end_dx.id == 2049 or @currentreport.end_dx.id == 252)
					r.correct_sentences << "0"
					r.correct_sentences << "1"
					r.correct_sentences << "2"
					r.correct_sentences << "3"
					r.missing_sentences << "4"
					r.missing_sentences << "5"
					r.diagnosis_found = true
				else
					r.correct_sentences << "0"
					r.correct_sentences << "1"
					r.missing_sentences << "2"
					r.missing_sentences << "3"
					r.missing_sentences << "4"
					r.diagnosis_found = false
				end

				@percentage = ((r.correct_sentences.length).to_f/(r.correct_sentences.length + r.missing_sentences.length))*100
				r.score = @percentage
			end
			####################################################################################
			# TEST CODE END #
			###################################################################################

			# Save student report
			r.save
			@studentreport = r

			#Calculate results from 5 most recent reports (for skill meters, graph)
			@recentfive = StudentReport.where(:learner_dx_id => @learnerdx.id, :user_id => current_user.id).order("created_at desc").limit(5)
			@recent_incorrect = 0
			@recent_correct = 0
			@recent_excellent = 0
			@recentfive.each do |f|
				if f.diagnosis_found == true
					if f.score > 70
						@recent_excellent += 0.2
					else
						@recent_correct += 0.2
					end
				else
						@recent_incorrect += 0.2
				end
			end

			@learnerdx.recent_incorrect = @recent_incorrect
			@learnerdx.recent_correct = @recent_correct
			@learnerdx.recent_excellent = @recent_excellent

			# Check if Level 1, 2 and 3 exist for the diagnosis in the learner model - if not, create them
			@enddx = EndDx.where(:id => @learnerdx.end_dx_id).first
			@dxlevel3 = "NOEXIST"
			@dxlevel2 = "NOEXIST"
			@dxlevel1 = "NOEXIST"
			if @enddx.dxable_type == "DxLevel3"
				@dxlevel3 = DxLevel3.where(:id => @enddx.dxable_id).first
				@dxlevel2 = DxLevel2.where(:id => @dxlevel3.dx_level2_id).first
				@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
			elsif @enddx.dxable_type == "DxLevel2"
				@dxlevel2 = DxLevel2.where(:id => @enddx.dxable_id).first
				@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
			else		
				@dxlevel1 = DxLevel1.where(:id => @enddx.dxable_id).first
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
				@currentreport.correct_dx += 1
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
			else
				@currentreport.incorrect_dx += 1
				@learnerdx.missed_dx += 1
			end

			@learnerdx.accuracy = @learnerdx.correct_dx/@learnerdx.cases_attempted.to_f
			@currentreport.difficulty = @currentreport.incorrect_dx/@currentreport.times_attempted.to_f
			@currentreport.save
			@learnerdx.save
			@learner_l1.save
			if @learner_l2.present?
				@learner_l2.save
			end
			if @learner_l3.present?
				@learner_l3.save
			end
			
			
		else
			# If user goes directly to 'practise cases' then it will just show stuff from the previous session
			@ids = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@currentreport = ExpertReport.find(@ids.sample)
			@learnerinfo.expert_report_id = @currentreport.id
			@learnerinfo.save
		end
			
		@showdx = EndDx.where(:id => @currentreport.end_dx_id).first
	end
end
