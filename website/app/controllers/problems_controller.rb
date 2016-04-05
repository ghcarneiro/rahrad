class ProblemsController < ApplicationController
    def index
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
end
