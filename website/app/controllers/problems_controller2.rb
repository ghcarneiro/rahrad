class ProblemsController < ApplicationController
    def index
		@ldx = LearnerDx.where(:review_list => true)
    end
	def user_select
		@dxlevel1s = DxLevel1.all
		if params[:search]
			@dx1 = DxLevel1.search(params[:search]).order("created_at DESC")
			@dx2 = DxLevel2.search(params[:search]).order("created_at DESC")
			@dx3 = DxLevel3.search(params[:search]).order("created_at DESC")
			@endx = EndDx.search(params[:search]).order("created_at DESC")
		end
			
	end

	def review_list
		@ldx = LearnerDx.where(:review_list => true)
	end

	def review_list_add
		@count = 0 
		@noparam = false
		@nofound = false
		if params[:search]
			if params[:search] == ""
				@noparam = true
			else
				@searchdx = LearnerDx.where(["name LIKE ?", "%#{params[:search]}%"]).where(:review_list => false).where(:user => current_user.id)
				#@searchdx = LearnerDx.search(params[:search]).order("created_at DESC")
				@already_reviewed = LearnerDx.where(["name LIKE ?", "%#{params[:search]}%"]).where(:review_list => true).where(:user => current_user.id)
			end
			if !@searchdx.present?
				@nofound = true
			end
		end

		if params[:id]
			@string = LearnerDx.where(:id => params[:id], :user => current_user.id)
			@string.each do |s|
				s.review_list = true
				s.save
				@count = @count + 1
			end
		end
		@ldx = LearnerDx.where(:review_list => true)
			
	end

	def review_list_remove
		@count = 0 # Counts the number of diagnoses selected to determine whether the singular or plural of diagnosis should be used
		@nodx = true # If no diagnoses have been selected for removal, no additional text will be displayed
		# Without this 'if' statement, the page will not load if no parameters have been passed
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

	def review_select
		@gensim = false
		@nodx = true # If no diagnoses have been selected for removal, no additional text will be displayed
		@learnerinfo = LearnerInfo.where(:user_id => current_user.id).first

		# All the expert reports from each selected diagnosis are connected 
		# to the user's learner info. A random report can then be selected out of all of these.
		# This code runs when the trainee selects diagnoses to review from their review list
		# It connects each report from each selected diagnosis to the trainee's learner info then chooses a random one to display
		if params[:id]
			# Remove old current report
			@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
			if @currentreport.present?
				@currentreport = "0"
			end

			@nodx = false
			@string = LearnerDx.where(:id => params[:id])
			@string.each do |n|
				@select = ExpertReport.where(:end_dx_id => n.end_dx_id)
				@select.each do |s|
					s.learner_info_id = @learnerinfo.id
					s.save
				end
			end
			@ids = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@currentreport = ExpertReport.find(@ids.sample)
			@learnerinfo.expert_report_id = @currentreport.id
			@learnerinfo.save

			#@ids = ExpertReport.where(:end_dx_id => @string.end_dx_id).pluck(:id)
			#@test = ExpertReport.find(@ids.sample)
			#@learnerinfo.expert_report_id = @test.id
			#@learnerinfo.save


		# create a file called fileName that will store information that will be used by the search engine
		# the search engine program will reference the name 'fileName' statically (because I don't know how to pass parameter using %x)
		# so you shouldn't change this name
		# This code runs when the trainee's report is submitted
		elsif params[:q].present? and @gensim = true
			@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
			@user_report = params[:q]

			fileName = "fileName"
			out_file = File.new(fileName, "w")
			# Get the original report, store in file
			out_file.puts(@currentreport.report_text)

			# 1) search query
			# Get search query, stores the search query in the file
			@query = params[:q]
			out_file.puts(@query)

			# done close the file
			out_file.close
			
			resultTemp = %x(python mock-similarity.py fileName)
			
			# we should process resultCSV so that it can be displayed correctly
			resultTemp = resultTemp.split("\n")
			@result = {
				n: Array.new,
				e: Array.new,
				m: Array.new,
				t: Array.new
			}	

			r = StudentReport.new
			r.diagnosis_found = true
			r.expert_report_id = @currentreport.id
			r.user_id = current_user.id
			r.report_sentences = Array.new
			@learnerdx = LearnerDx.where(:end_dx_id => @currentreport.end_dx_id).first
			r.learner_dx_id = @learnerdx.id

			@createnew = 0

			resultTemp.each do |i|
				if (i.include? "n	") or (i.include? "e	") or (i.include? "m	") or (i.include? "t	")
					temp = i.split("\t")
					@result[temp[0].to_sym].push(temp[1])
					if temp[0].to_sym == :"n"
						r.correct_sentences << temp[2]
						r.report_sentences << temp[1]
					elsif temp[0].to_sym == :"e"
						r.report_sentences << temp[1]
					elsif temp[0].to_sym == :"m"
						r.missing_sentences << temp[2]
						if @currentreport.report_sentences.nil?
							@currentreport.report_sentences = Array.new
							@currentreport.report_sentences << temp[1]
							@createnew = 1
						elsif @createnew = 1
							@currentreport.report_sentences << temp[1]
						end
					elsif temp[0].to_sym == :"t"
						if @currentreport.report_sentences.nil?
							@currentreport.report_sentences << temp[1]
							@createnew = 1
						elsif @createnew = 1
							@currentreport.report_sentences << temp[1]
						end
					end
				end
			end

			# Save student report
			r.save
			@studentreport = r
			@studenttext = r.report_sentences
			@experttext = @currentreport.report_sentences
			@percentage = ((@result[:n].length).to_f/(@result[:m].length + @result[:n].length))*100

		elsif params[:q].present? and @gensim = false
			@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
			@user_report = params[:q]

			fileName = "fileName"
			out_file = File.new(fileName, "w")
			# Get the original report, store in file
			out_file.puts(@currentreport.report_text)

			# 1) search query
			# Get search query, stores the search query in the file
			@query = params[:q]
			out_file.puts(@query)

			# done close the file
			out_file.close
			
			resultTemp = %x(python random-feedback.py fileName)
			
			# we should process resultCSV so that it can be displayed correctly
			resultTemp = resultTemp.split("\n")
			@result = {
				n: Array.new,
				e: Array.new,
				m: Array.new,
				t: Array.new
			}	

			r = StudentReport.new
			r.diagnosis_found = true
			r.expert_report_id = @currentreport.id
			r.user_id = current_user.id
			@learnerdx = LearnerDx.where(:end_dx_id => @currentreport.end_dx_id).first
			r.learner_dx_id = @learnerdx.id

			@createnew = 0

			resultTemp.each do |i|
				if (i.include? "n	") or (i.include? "e	") or (i.include? "m	") or (i.include? "t	")
					temp = i.split("\t")
					@result[temp[0].to_sym].push(temp[1])
					if temp[0].to_sym == :"n"
						r.correct_sentences << temp[2]
						r.report_sentences << temp[1]
					elsif temp[0].to_sym == :"e"
						r.report_sentences << temp[1]
					elsif temp[0].to_sym == :"m"
						r.missing_sentences << temp[2]
						if @currentreport.report_sentences.nil?
							@currentreport.report_sentences << temp[1]
							@createnew = 1
						elsif @createnew = 1
							@currentreport.report_sentences << temp[1]
						end
					elsif temp[0].to_sym == :"t"
						if @currentreport.report_sentences.nil?
							@currentreport.report_sentences << temp[1]
							@createnew = 1
						elsif @createnew = 1
							@currentreport.report_sentences << temp[1]
						end
					end
				end
			end

			# Save student report
			r.save
			@studentreport = r
			@studenttext = r.report_sentences
			@experttext = @currentreport.report_sentences
			@percentage = ((@result[:n].length).to_f/(@result[:m].length + @result[:n].length))*100
	

			# Then we deleted the fileName file, because the search_engine program finished using it
			#File.delete(fileName)
		# Second report
		else
			@ids = ExpertReport.where(:learner_info_id => @learnerinfo.id)
			@currentreport = ExpertReport.find(@ids.sample)
			@learnerinfo.expert_report_id = @currentreport.id
			@learnerinfo.save
		end
	end
end
