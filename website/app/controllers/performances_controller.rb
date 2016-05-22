class PerformancesController < ApplicationController
	def index



	end

def concept
	    @pagetype = "concept"
	    @conceptlist = "public/conceptlist_" + current_user.id.to_s
	    @dxlevel1s = DxLevel1.all
	    @learner_level1s = LearnerLevel1.where(:user_id => current_user.id)
	    @learner_level2s = LearnerLevel2.where(:user_id => current_user.id)
	    @learner_level3s = LearnerLevel3.where(:user_id => current_user.id)
	    @learner_dxes = LearnerDx.where(:user_id => current_user.id)
            #@nodxlevel3s = DxLevel2.includes(:dx_level3s).where(:dx_level3s => {:dxable_id => nil})
	    #@hasdxlevel3s = DxLevel2.includes(:dx_level3s).where("dx_level3s.dxable_id IS NOT NULL")

end

# Add to review list
def add
	    if params[:l]
	    @l_id = params[:l]
	    @add_dx = LearnerDx.where(:id => @l_id, :user_id => current_user.id).first
	    @add_dx.review_list = true
	    @add_dx.save
	    
	    @html = '<span class="key">Added to review list</span><br/><span class="btn btn-danger btn-sm"><span class="glyphicon glyphicon-minus"></span> Remove from review list</span>'
	    end

	    # Concept hierarchy
	    if params[:c]
	    @e_id = params[:c]
	    @add_dx = LearnerDx.where(:end_dx_id => @e_id, :user_id => current_user.id).first
	    @e = EndDx.where(:id => @e_id).first
	    # CREATE LEARNER DX IF DOES NOT EXIST
	    if @add_dx.nil?
					@add_dx = LearnerDx.new
					@add_dx.name = @e.name
					@add_dx.end_dx_id = @e.id
					@add_dx.user_id = current_user.id
					@add_dx.cases_attempted = 0
					@add_dx.missed_dx = 0
					@add_dx.accuracy = 0
					@add_dx.correct_dx = 0
					@add_dx.excellent_cases = 0
	    end
	    @add_dx.review_list = true
	    @add_dx.save
	    @add = LearnerDx.where(:end_dx_id => @e_id, :user_id => current_user.id).first

	    		    @correct = @add.correct_dx/@add.cases_attempted.to_f
		    @excellent = @add.excellent_cases/@add.cases_attempted.to_f
			if @excellent > 0.5
			    @html = "<img height='15' width='15' src='/assets/green-tick.gif'></img>"
			elsif @correct > 0.5
			    @html = "<img height='15' width='15' src='/assets/yellow-tick.gif'></img>"
			elsif @add.cases_attempted == 0
			    @html = "<img height='15' width='15' src='/assets/grey-tick.gif'></img>"
			else
			    @html = "<img height='15' width='15' src='/assets/red-tick.gif'></img>"
			end
	    end

	    respond_to do |format|
  	    	format.html do
		    render text: @html 
		end
	    end
end

# Remove from review list
def remove
	    if params[:l]
	    @l_id = params[:l]
	    @add_dx = LearnerDx.where(:id => @l_id, :user_id => current_user.id).first
	    @add_dx.review_list = false
	    @add_dx.save
	    
	    @html = '<span class="error">Removed from review list</span><br/><span class="btn btn-success btn-sm"><span class="glyphicon glyphicon-plus"></span> Add to review list</span>'
	    end

	    if params[:c]
	    @e_id = params[:c]
	    @remove_dx = LearnerDx.where(:end_dx_id => @e_id, :user_id => current_user.id).first
	    @remove_dx.review_list = false
	    @remove_dx.save
	    
	    	    		    @correct = @remove_dx.correct_dx/@remove_dx.cases_attempted.to_f
		    @excellent = @remove_dx.excellent_cases/@remove_dx.cases_attempted.to_f
			if @excellent > 0.5
			    @html = "<img height='15' width='15' src='/assets/green.gif'></img>"
			elsif @correct > 0.5
			    @html = "<img height='15' width='15' src='/assets/yellow.gif'></img>"
			elsif @remove_dx.cases_attempted == 0
			    @html = "<img height='15' width='15' src='/assets/grey.gif'></img>"
			else
			    @html = "<img height='15' width='15' src='/assets/red.gif'></img>"
			end
	    end

	    respond_to do |format|
  	    	format.html do
		    render text: @html 
		end
	    end
end

#Retrieve data for concept hierarchy
def data

	    @s = params[:l].split("_")
	    @level = @s[0]
	    @id = @s[1]
	    params[:search_id] = @id.to_i
	    params[:year_level] = current_user.year_of_training
	    @isend = 0
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
	    @next.each do |n|
		@html = @html + "<table><tr><td>"

		# Retrieve data from learner model
	    	if @level == "l1"
		    @ldx = LearnerLevel2.where(:dx_level2_id => n.id).where(:user_id => current_user.id).first
		    @i = "l2_" + n.id.to_s
		elsif @level == "l2"
		    if @isend == 1
		    	@ldx = LearnerDx.where(:end_dx_id => n.id).where(:user_id => current_user.id).first
		        @i = "e_" + n.id.to_s
		    else
			@ldx = LearnerLevel3.where(:dx_level3_id => n.id).where(:user_id => current_user.id).first
		    	@i = "l3_" + n.id.to_s
		    end
		elsif @level == "l3"
			@ldx = LearnerDx.where(:end_dx_id => n.id).where(:user_id => current_user.id).first
		        @i = "e_" + n.id.to_s
		end


		if !@i.include?("e_")
		if (@ldx.nil?) or (@ldx.cases_attempted == 0)
		    @html = @html + "<img src='/assets/grey.gif' width='15' height='15' /> "
		else
		    @correct = @ldx.correct_dx/@ldx.cases_attempted.to_f
		    @excellent = @ldx.excellent_cases/@ldx.cases_attempted.to_f
			if @excellent > 0.5
		    	    @html = @html + "<img src='/assets/green.gif' width='15' height='15' /> "
			elsif @correct > 0.5
		    	    @html = @html + "<img src='/assets/yellow.gif' width='15' height='15' /> "
			else
		    	    @html = @html + "<img src='/assets/red.gif' width='15' height='15' /> "
			end
		end
  	    	@html = @html + "<span class='dx-toggle' id='" + @i + "'>" + n.name + "<span class='glyphicon glyphicon-menu-right'></span></span></td></tr></table>"
		else
	      @popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 50%; display: none">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: 50px">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + n.name + ': X% correct</span>
	      </div> '
		
		if n.category == "key"
			@categorytext = ' <strong class="key-strong">Key condition</strong>'
		elsif n.category == "1"
			@categorytext=' <strong>Category 1</strong>'
		elsif n.category == "2"
			@categorytext = ' <strong>Category 2</strong>'
		elsif n.category == "3"
			@categorytext= ' <strong>Category 3</strong>'
		end

		if (!@ldx.nil?)
		    @correct = @ldx.correct_dx/@ldx.cases_attempted.to_f
		    @excellent = @ldx.excellent_cases/@ldx.cases_attempted.to_f
		    if @ldx.review_list == true # Override default set above
			if @excellent > 0.5
			    @reviewhtml = "<span class='remove' id='" + n.id.to_s + "'><img src='/assets/green-tick.gif' width='15' height='15'></span> "
			elsif @correct > 0.5
			    @reviewhtml = "<span class='remove' id='" + n.id.to_s + "'><img src='/assets/yellow-tick.gif' width='15' height='15'></span> "
			elsif @ldx.cases_attempted == 0
			    @reviewhtml = "<span class='remove' id='" + n.id.to_s + "'><img src='/assets/grey-tick.gif' width='15' height='15'></span> "
			else
			    @reviewhtml = "<span class='remove' id='" + n.id.to_s + "'><img src='/assets/red-tick.gif' width='15' height='15'></span> "
			end
		    else
			if @excellent > 0.5
			    @reviewhtml = "<span class='add' id='" + n.id.to_s + "'><img src='/assets/green.gif' width='15' height='15'></span> "
			elsif @correct > 0.5
			    @reviewhtml = "<span class='add' id='" + n.id.to_s + "'><img src='/assets/yellow.gif' width='15' height='15'></span> "
			elsif @ldx.cases_attempted == 0
			    @reviewhtml = "<span class='add' id='" + n.id.to_s + "'><img src='/assets/grey.gif' width='15' height='15'></span> "
			else
			    @reviewhtml = "<span class='add' id='" + n.id.to_s + "'><img src='/assets/red.gif' width='15' height='15'></span> "
			end
		    end
		else
		    @reviewhtml = "<span class='add' id='" + n.id.to_s + "'><img src='/assets/grey.gif' width='15' height='15'></span> "
		end

  	    	@html = @html + "<table><tr><td>"  + @reviewhtml + "<span class='endDx'>" + n.name + @categorytext + "</span>" + @popup + "</td></tr></table>"	
		end
	    end

	    # Add key diagnoses if level 1
 	    if @level == "l1" and (current_user.year_of_training == "1" or current_user.year_of_training == "2")
	    	@keydx.each do |k|
		@ldx = LearnerDx.where(:end_dx_id => k.id).where(:user_id => current_user.id).first
	      @popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 50%; display: none">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: 50px">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + k.name + ': X% correct</span>
	      </div>'
		if (!@ldx.nil?)
		    @correct = @ldx.correct_dx/@ldx.cases_attempted.to_f
		    @excellent = @ldx.excellent_cases/@ldx.cases_attempted.to_f
		    if @ldx.review_list == true # Override default set above
			if @excellent > 0.5
			    @reviewhtml = "<span class='remove' id='" + k.id.to_s + "'><img src='/assets/green-tick.gif' width='15' height='15'></span> "
			elsif @correct > 0.5
			    @reviewhtml = "<span class='remove' id='" + k.id.to_s + "'><img src='/assets/yellow-tick.gif' width='15' height='15'></span> "
			elsif @ldx.cases_attempted == 0
			    @reviewhtml = "<span class='remove' id='" + k.id.to_s + "'><img src='/assets/grey-tick.gif' width='15' height='15'></span> "
			else
			    @reviewhtml = "<span class='remove' id='" + k.id.to_s + "'><img src='/assets/red-tick.gif' width='15' height='15'></span> "
			end
		    else
			if @excellent > 0.5
			    @reviewhtml = "<span class='add' id='" + k.id.to_s + "'><img src='/assets/green.gif' width='15' height='15'></span> "
			elsif @correct > 0.5
			    @reviewhtml = "<span class='add' id='" + k.id.to_s + "'><img src='/assets/yellow.gif' width='15' height='15'></span> "
			elsif @ldx.cases_attempted == 0
			    @reviewhtml = "<span class='add' id='" + k.id.to_s + "'><img src='/assets/grey.gif' width='15' height='15'></span> "
			else
			    @reviewhtml = "<span class='add' id='" + k.id.to_s + "'><img src='/assets/red.gif' width='15' height='15'></span> "
			end
		    end
		else
		    @reviewhtml = "<span class='add' id='" + k.id.to_s + "'><img src='/assets/grey.gif' width='15' height='15'></span> "
		end

		@html = @html + "<table><tr><td>" + @reviewhtml + "<span class='endDx'>" + k.name + " <strong class='key-strong'>Key condition</strong></span>" + @popup + "</td></tr></table>"
	    	end
	    end

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

def list
	    @pagetype = "list"
	    @level = 1
	    @learner_level1s = LearnerLevel1.where(:user_id => current_user.id)	
	    # Display the appropriate level 2 data for the selected level 1 diagnosis  
	    if params[:l1]  
		@level = 2
		# Get level 2 diagnoses
		@learner1 = LearnerLevel1.where(:id => params[:l1], :user_id => current_user.id).first
		@dxlevel1 = DxLevel1.where(:id => @learner1.dx_level1_id).first
		@dxlevel2s = DxLevel2.where(:dx_level1_id => @dxlevel1.id)

		# Get key conditions
		@keyconditions = EndDx.where(:dxable_type => "DxLevel1").where(:dxable_id => @dxlevel1.id)
	    end
	    if params[:l2]  
		@level = 3
		# Get level 3 diagnoses
		@learner2 = LearnerLevel2.where(:id => params[:l2], :user_id => current_user.id).first
		@dxlevel2 = DxLevel2.where(:id => @learner2.dx_level2_id).first
		@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
		@dxlevel3s = DxLevel3.where(:dx_level2_id => @dxlevel2.id)
		if @dxlevel3s.empty?
		    @level = 4
		    params[:search_id] = @dxlevel2.id
		    params[:search_type] = "DxLevel2"
		    params[:year_level] = current_user.year_of_training
		    @end_dxes = EndDx.dxable_search3(params[:search_id], params[:search_type], params[:year_level])
		end
	    end
	    if params[:l3]  
		@level = 4
		# Get level 3 diagnoses
		@learner3 = LearnerLevel3.where(:id => params[:l3], :user_id => current_user.id).first
		@dxlevel3 = DxLevel3.where(:id => @learner3.dx_level3_id).first
		@dxlevel2 = DxLevel2.where(:id => @dxlevel3.dx_level2_id).first
		@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
		    params[:search_id] = @dxlevel3.id
		    params[:search_type] = "DxLevel3"
		    params[:year_level] = current_user.year_of_training
		@end_dxes = EndDx.dxable_search3(params[:search_id], params[:search_type], params[:year_level])
	    end
end

def missed_dx
	    @pagetype = "misseddx"
	    @missed = LearnerDx.where('missed_dx > ?', 0).where(:user_id => current_user.id).order('accuracy asc, cases_attempted desc')
end

def report_hx
	    @pagetype = "reporthx"
	    @pageno = 1
	    @offset = 0
	    @student_reports = ((StudentReport.where(:user_id => current_user.id)).sort_by &:created_at).reverse
	    @length = @student_reports.length
	    if @length > 10
		@pageno = @length/10
		if @length > @pageno*10
		    @pageno = @pageno + 1
		end
	    end

	    @studentreports = ((StudentReport.where(:user_id => current_user.id).limit(10)).sort_by &:created_at).reverse

	    if params[:id]
	    	@currentreport = StudentReport.where(:id => params[:id]).first
		@expertreport = ExpertReport.where(:id => @currentreport.expert_report_id).first
		@expert_sentences = @expertreport.report_text.split(".")
		@student_sentences = @currentreport.report_text.split(".")
	    end
	    if params[:count]
		@offset = @pageno - params[:count].to_i
		@studentreports = ((StudentReport.where(:user_id => current_user.id).offset(@offset*10).limit(10)).sort_by &:created_at).reverse

	    end

end

def skillmeters
	    @pagetype = "skillmeters"
	    @l1 = LearnerLevel1.where(:user_id => current_user.id)	
end

end
