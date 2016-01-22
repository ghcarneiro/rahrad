class GraphController < ApplicationController

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
		# I only want date, because to display the graph I only need dates
		out_file.puts("onlyreturndates")

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
		@resultCSV = %x(python search_engine.py fileName)
		
		# Then we deleted the fileName file, because the search_engine program finished using it
		File.delete(fileName)

		# prepare chart data that will be used to generate chart later: step 1 - get all the years of the report
		# and put those years into an array called reportDates
		# reportDates will become an array of integer, each integer represent a report's year
		reportDates = Array.new
		@resultCSV.each_line do |line|
			# the first line is header, so ignore it
			if line != @resultCSV.lines.first
				line.split("<,>").each do |f|
					if f == line.split("<,>")[1]	#[1] means second element, which is the date column
		 				reportDates.push(f[0..3].to_i)	#[6...9] means get substring of f, which will be the year part
		 			end
				end
			end
		end

		# prepare chart data that will be used to generate chart later: step 2 - count occurence of identical year
		# counts is a hash, it contains the occurence of each year in the reportDates array
		counts = Hash.new 0
		reportDates.each do |reportDate|
			counts[reportDate] += 1
		end
		
		# if I am going to plot on Google chart using data above, I need all year for the graph to look correct
		# Let me explain, I need to have all year, even those year that has occurence of zero because my Google charts
		# only prints what I gave to it, if I have 2000, once, 2002 twice as data then it will not display 2001 zero times
		# which will look strange
		# Now, sort the hash 'counts', so later on it will be easier to fill in the missing year
		counts = counts.sort.to_h

		# get the max year occurence count, [1] returns the value part of this max hash
		@maxYearCount = counts.max_by{|k,v| v}[1]

		# Fill in the missing years
		firstYear = counts.keys.first	#get the first existing year
		lastYear = counts.keys.last		#get the last existing year
		# create a new hash from first to last existing year, each hash item with zero occurence
		counts2 = Hash.new 0
		for year in firstYear..lastYear
			counts2[year] += 0
		end
		# merge the two hases, so now all the missing years will be filled
		counts3 = counts2.merge(counts)

		# turn hash into string that can be process by graph page Google chart Javascript
		final = ""
		counts3.each do |key, value|
			final += key.to_s + "," + value.to_s + ","
		end
		final = final.chomp(',')	#remove the last comma

		# put this string in a variable that can be accessed in the graph view page
		@resultCSV = final

		# When I perform this index action, I may want to look at the html page or pdf page
		respond_to do |format|
			format.html
			format.pdf do

				# If I am going to see the pdf, then I need to generate charts and pdf
				# I did not use Google chart to generate chart in my pdf, because I couldn't get that to work
				# Instead, I use gruff gem to generate gruff, and use prawn gem to generate pdf
				# I first save the gruff charts into server local folder, then I load those image using prawn

				# To make gruff chart, I need to prepare data for gruff to use
				# prepare the x/horizontal axis label, gruff take laebl as a hash, this axis will be the years
				# for detail, read gruff documentation
				# This label is for the gruff line chart
				label_hash = Hash.new 0
				i = 0
				for year in firstYear..lastYear
					if year%2 != 0 	# x axis is too crowdy, cut crowd in half
						label_hash[i] = year.to_s
					else
						label_hash[i] = " "
					end
					i += 1
				end

				# Line chart setting
				g = Gruff::Line.new(1920)
				g.title = 'Number of matching reports per year'
				g.line_width = 1
				g.dot_radius = 3
				g.hide_legend = true
				g.title_font_size = 22 # Title font size
				g.theme = {
				  :colors => ['orange'],
				  :marker_color => 'grey',
				  :font_color => 'black',
				  :background_colors => 'transparent'
				}
				g.labels = label_hash
				# gruff takes data as an array so I converted it here
				g.data(:reports, counts3.values)
				# produce the image of the chart and save it to server side local folder
				g.write("#{Rails.root}/app/assets/images/line_chart.png")

				# Pie chart
				g = Gruff::Pie.new(1200)
				g.title = 'Number of matching reports per year'
				g.title_font_size = 22 # Title font size
				g.text_offset_percentage = 0.05
				g.theme = {
				  :colors => ['#fcc39c', '#5574A6', '#329262', '#8B0707', '#6633CC', '#AAAA11', '#22AA99', '#994499', '#316395', '#B82E2E', '#66AA00', '#DD4477', '#0099C6', '#3B3EAC', '#990099', '#109618', '#FF9900', '#DC3912', '#008744', '#0057e7', '#d62d20', '#ffa700'],
				  :marker_color => 'grey',
				  :font_color => 'black',
				  :background_colors => 'transparent'
				}

				# give gruff the label and data
				# For pie chart, I don't want to display any zero occurence year
				counts3.each do |key, value|
					if value > 0
						g.data key.to_s, value
					end
				end
				g.write("#{Rails.root}/app/assets/images/pie_chart.png")


				# Prepar new labels for bar chart
				i = 0
				for year in firstYear..lastYear
					label_hash[i] = year.to_s
					i += 1
				end

				# Bar chart
				g = Gruff::SideBar.new(1920)
				g.title = 'Number of matching reports per year'
				g.hide_legend = true
				g.title_font_size = 22 # Title font size
				g.bar_spacing = 0.5
				g.theme = {
				  :colors => ['#5bc0de'],
				  :marker_color => 'grey',
				  :font_color => 'black',
				  :background_colors => 'transparent'
				}
				g.labels = label_hash
				# bar chart takes array of data
				g.data(:reports, counts3.values)
				g.write("#{Rails.root}/app/assets/images/bar_chart.png")

				# Create new prawn pdf
				pdf = Prawn::Document.new(:page_size => "A4", :page_layout => :landscape)
				# place the chart images in the pdf
				pdf.image "#{Rails.root}/app/assets/images/line_chart.png", :width => 765, :height => 520
				pdf.image "#{Rails.root}/app/assets/images/pie_chart.png", :width => 765, :height => 520
				pdf.image "#{Rails.root}/app/assets/images/bar_chart.png", :width => 765, :height => 520
				send_data pdf.render, filename: "analysis.pdf", type: "application/pdf", disposition: "inline"
			end
		end
	end
end
