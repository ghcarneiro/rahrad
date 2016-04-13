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
		@nodx = true # If no diagnoses have been selected for removal, no additional text will be displayed
		@learnerinfo = LearnerInfo.where(:user_id => current_user.id).first
		if params[:id]
			@nodx = false
			@string = LearnerDx.where(:id => params[:id])			
			@string.each do |s|
				@test = ExpertReport.where(:end_dx_id => s.end_dx_id).first
				@learnerinfo.expert_report_id = @test.id
				@learnerinfo.save
			end
		end


		@currentreport = ExpertReport.where(:id => @learnerinfo.expert_report_id).first
		# create a file called fileName that will store information that will be used by the search engine
		# the search engine program will reference the name 'fileName' statically (because I don't know how to pass parameter using %x)
		# so you shouldn't change this name
		if params[:q].present?
			@user_report = params[:q]
			fileName = "fileName"
			out_file = File.new(fileName, "w")
			# Get the original report, store in file
			out_file.puts('"00R000062","2400      CT HEAD - PLAIN L3   CT HEAD CLINICAL DETAILS:  UNENHANCED AXIAL IMAGES FROM SKULL BASE TO VERTEX.  IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  CLINICAL DETAILS:  MVA ROLL OVER.  AMNESIC TO EVENT.  GCS 15 AT SCENE - LOC IN TRANSIT.   NOW 13-15, INAPPROPRIATE.  LARGE RIGHT TEMPORAL HAEMTOMA. MALOCCLUSION OF JAW.  FINDINGS:  RIGHT SIDED SCALP HAEMATOMA.  NO EXTRA-AXIAL COLLECTION.  THE GREY WHITE MATTER DIFFERENTIATION IS NORMAL.  NO HAEMORRHAGE OR MASS LESION.  NO EVIDENCE OF ACUTE OR OLD INFARCTION.  THE DILATATION OF THE VENTRICLES, SULCI AND BASAL CISTERNS ARE NORMAL FOR THE PATIENT\'S AGE.  NO FRACTURE DEMONSTRATED. CONCLUSION:  RIGHT SIDED SCALP HAEMATOMA.  OTHERWISE NORMAL STUDY')

			# 1) search query
			# Get search query, stores the search query in the file
			@query = params[:q]
			threshold = params[:t].to_f
			modelType = params[:modelType]
			puts "model Type"
			puts modelType
			out_file.puts(@query)

			# done close the file
			out_file.close
			
			resultTemp = %x(python similarity.py fileName #{threshold} #{modelType})
			
			# we should process resultCSV so that it can be displayed correctly
			resultTemp = resultTemp.split("\n")
			@result = {
				n: Array.new,
				e: Array.new,
				m: Array.new
			}	
			resultTemp.each do |i|
				temp = i.split("\t")
				@result[temp[0].to_sym].push(temp[1])
			end	
			# Then we deleted the fileName file, because the search_engine program finished using it
			#File.delete(fileName)
		end
	end
end
