class TeachingsController < ApplicationController

	def index
		# create a file called fileName that will store information that will be used by the search engine
		# the search engine program will reference the name 'fileName' statically (because I don't know how to pass parameter using %x)
		# so you shouldn't change this name
		if params[:q].present?
			
			fileName = "fileName"
			out_file = File.new(fileName, "w")
			# Get the original report, store in file
			out_file.puts('"00R000062","2400      CT HEAD - PLAIN L3   CT HEAD CLINICAL DETAILS:  UNENHANCED AXIAL IMAGES FROM SKULL BASE TO VERTEX.  IMAGES PHOTOGRAPHED ON SOFT TISSUE AND BONE WINDOWS.  CLINICAL DETAILS:  MVA ROLL OVER.  AMNESIC TO EVENT.  GCS 15 AT SCENE - LOC IN TRANSIT.   NOW 13-15, INAPPROPRIATE.  LARGE RIGHT TEMPORAL HAEMTOMA. MALOCCLUSION OF JAW.  FINDINGS:  RIGHT SIDED SCALP HAEMATOMA.  NO EXTRA-AXIAL COLLECTION.  THE GREY WHITE MATTER DIFFERENTIATION IS NORMAL.  NO HAEMORRHAGE OR MASS LESION.  NO EVIDENCE OF ACUTE OR OLD INFARCTION.  THE DILATATION OF THE VENTRICLES, SULCI AND BASAL CISTERNS ARE NORMAL FOR THE PATIENT\'S AGE.  NO FRACTURE DEMONSTRATED. CONCLUSION:  RIGHT SIDED SCALP HAEMATOMA.  OTHERWISE NORMAL STUDY')

			# 1) search query
			# Get search query, stores the search query in the file
			@query = params[:q]
			out_file.puts(@query)

			# done close the file
			out_file.close

			resultTemp = %x(python similarity.py fileName)
			
			# we should process resultCSV so that it can be displayed correctly
			resultTemp = resultTemp.split("\n")
			@result = []	
			resultTemp.each do |i|
				temp = i.split("\t")
				if temp[0] == "n"	
					@result.push(["n",temp[1]])
				elsif temp[0] == "c"
					@result.push(["c",temp[1],temp[2]])
				elsif temp[0] == "m"
					@result.push(["m",temp[1]])
				end
			end	
			# Then we deleted the fileName file, because the search_engine program finished using it
			#File.delete(fileName)
		end
	end

	def save_search
		# save the search result as a saved_search object for this user
		query = params[:query]
		time = params[:time]
		current_user.saved_searches.create(:query => query, :time => time)
		
		# stay in the same page and display message saying that the search has been saved
		flash[:notice] = 'Search saved.'
		redirect_to :back
	end

	def display_report
		# prepare the report accession number, date and content that will be displayed in the view page later
		reportNumberDateContent = params[:reportNumberDateContent]	#the search result
		@accessionNumber = reportNumberDateContent.split("<,>")[0]
		date = reportNumberDateContent.split("<,>")[1]
		@year = date[0..3]
		@month = date[4..5]
		@day = date[6..7]
		@content = reportNumberDateContent.split("<,>")[2]
	end

end
