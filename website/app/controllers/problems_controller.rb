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
end
