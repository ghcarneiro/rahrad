class SavedSearchController < ApplicationController
	def index
		# get the saved searches and reoccurences of this user, it will be displayed
		@savedSearches = current_user.saved_searches
		@reoccurences = current_user.reoccurences

		# For each reoccurence, since we do not actaully repeat the search periodically,
		# everytime the  user visit this page we need to create those missing saved search
		# create all the missing savedsearch according to the reoccurence and existing savedSearches
		today = DateTime.now.to_date
		@reoccurences.each do |reoccurence|
			# if the unit is day
			if reoccurence.unit.downcase == "day" || reoccurence.unit.downcase == "days"
				# from the time this reoccurence is last updated to today
				for date in reoccurence.time.to_date..today
					# if it is a date that the search should be repeated and saved,
					# and that day is not the last updated date for the reoccurence
					if (date - reoccurence.time.to_date)%reoccurence.interval == 0 && date != reoccurence.time.to_date
						# check if this search is already saved 
						saved = false
						@savedSearches.each do |savedSearch|
							if savedSearch.time.to_date == date
								saved = true
							end
						end
						# if this search is not already saved, then create a new saved_searches and save it
						if !saved
							current_user.saved_searches.create(:query => reoccurence.query, :time => DateTime.parse(date.to_s))
						end
					end
				end
			end

			# if the unit is week
			if reoccurence.unit.downcase == "week" || reoccurence.unit.downcase == "weeks"
				# from the time this reoccurence is last updated to today
				for date in reoccurence.time.to_date..today
					# if it is a date that the search should be repeated and saved,
					# and that day is not the last updated date for the reoccurence
					if (date - reoccurence.time.to_date)%(7*reoccurence.interval) == 0 && date != reoccurence.time.to_date
						# check if this search is already saved 
						saved = false
						@savedSearches.each do |savedSearch|
							if savedSearch.time.to_date == date
								saved = true
							end
						end
						# if this search is not already saved, then create a new saved_searches and save it
						if !saved
							current_user.saved_searches.create(:query => reoccurence.query, :time => DateTime.parse(date.to_s))
						end
					end
				end
			end

			# if the unit is month
			# month is different, if the user start the reoccurence at 31st of some month, 
			# then next month may not have the 31st, in that case I will save the search at the last day: 28th, 29th or 30th
			if reoccurence.unit.downcase == "month" || reoccurence.unit.downcase == "months"
				date = reoccurence.created_at.to_date
				c = 1
				# if it is a date that the search should be repeated and saved,
				# and that day is not the last updated date for the reoccurence
				while date>>c <= DateTime.now.to_date && (date>>c)>reoccurence.time.to_date
					# check if this search is already saved 
					saved = false
					@savedSearches.each do |savedSearch|
						if savedSearch.time.to_date == date>>c
							saved = true
						end
					end
					# if this search is not already saved, then create a new saved_searches and save it
					if !saved
						current_user.saved_searches.create(:query => reoccurence.query, :time => DateTime.parse((date>>c).to_s))
					end
					c+=1
				end
			end

			# update the last updated date for this reoccurence
			reoccurence.time = DateTime.new(DateTime.now.year,DateTime.now.month,DateTime.now.day)				
			
		end
	end

	# destroy saved search
	def destroy
		@saved_search = SavedSearch.find(params[:id])
	    @saved_search.destroy

	    #stay in the same page
		redirect_to :back
	end

	# destroy reoccurence
	def destroy_reoccurence
		@reoccurence = Reoccurence.find(params[:id])
	    @reoccurence.destroy

	    #stay in the same page
		redirect_to :back
	end
	
	# Go to the edit_reoccurence page which can be used to wither create or edit reoccurence
	def edit_reoccurence
		# title of edit_reoccurence page that will be displayed
		@title = params[:title]
		# below will be auto filled in the edit_reoccurence page.
		# If absent then the corresponding box in edit_reoccurence will be blank
		@query = params[:query]
		@interval = params[:interval]
		@unit = params[:unit]
		# the id of the reoccurence, used in editing_reoccurence method below
		@id = params[:id]
	end
	def editing_reoccurence
		# If you are not editing reoccurence but creating reoccurence then don't destroy anything
		if params[:id].present?
			@reoccurence = Reoccurence.find(params[:id])
		    @reoccurence.destroy
		end

		#create and save that new reoccurences
		query = params[:query]
		interval = params[:interval]
		unit = params[:unit]
		time = DateTime.now
		current_user.reoccurences.create(:query => query, :interval => interval, :unit => unit, :time => DateTime.now)

		# redirect back to the saved search and reoccurences page,
		redirect_to action: "index"
	end
end
