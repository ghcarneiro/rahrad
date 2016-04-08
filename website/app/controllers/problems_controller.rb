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
		if params[:search]
			@endx = EndDx.search(params[:search]).order("created_at DESC")
		end
			
	end

	def review_list_add

		@string = LearnerDx.where(:end_dx_id => params[:end_dx][:id], :user => current_user.id)
		@string.each do |s|
			s.review_list = true
			s.save
		end
		@ldx = LearnerDx.where(:review_list => true)

		
			
	end
end
