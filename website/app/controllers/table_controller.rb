class TableController < ApplicationController

	def index
		# create a file called fileName that will store information that will be used by the search engine
		# the search engine program will reference the name 'fileName' statically (because I don't know how to pass parameter using %x)
		# so you shouldn't change this name
		fileName = "fileName"
		out_file = File.new(fileName, "w")

		# 1) search query
		# Get search query, stores the search query in the file
		@query = params[:q]
		out_file.puts(@query)

		# 2) do I only wnat the search engine to return me accession number and dates of the reports?
		# tell the search engine whether you only want dates and the accesion number of report
		# I will only want date in the graph controller, because to display the graph I only need dates
		# But now I want the full results: report accesion number, report date, report content
		out_file.puts("notonlyreturndates")

		# 3) do I only want report before a specific time? 
		# if came from savedSearch, you may want to filter out results after certain DateTime, figure out the time here
		@time = DateTime.now.to_s
		if params[:time].present?
			@time = params[:time]
		end
		@time = @time.to_datetime
		month = @time.month
		day = @time.day
		if month < 10
			month = "0#{month}"
		end
		if day < 10
			day = "0#{day}"
		end
		# then the search engine shall only return you reports before 'queryTime' (inclusive)
		queryTime = "#{@time.year}#{@time.month}#{@time.day}"
		out_file.puts(queryTime)
		out_file.close

		# run the search engine: the search engine read the search terms by accessing the fileName file and print output to the standard output
		# we collect the standard output produced by the search engine using %x() and store it in a variable called @resultCSV
		# the output will be separated by <,>
		# In other words, the output is a csv file with the delimiter '<,>'
		# We use 3 character '<', ',' and  '>' because the hospital report may have comma in thier report content
		# and we do use this delimeter to take care of that
		# the search engine program shall be an executable file named search_engine in the 'dist' folder
		#@resultCSV = %x(./dist/search_engine fileName)
		@resultCSV = %x(python search_engine.py fileName)
		
		# Then we deleted the fileName file, because the search_engine program finished using it
		File.delete(fileName)

		# depending on how I access this index action, I want to open it as a html or csv
		respond_to do |format|
			format.html
			format.csv do
				send_data @resultCSV, :disposition => 'inline', :filename => 'csv_search_results.csv', type: Mime::CSV
			end
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
