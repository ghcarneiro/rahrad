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
					@add_dx.recent_excellent = 0
					@add_dx.recent_correct = 0
					@add_dx.recent_incorrect = 0
	   	end

	    	@add_dx.review_list = true
	    	@add_dx.save
	    	@add = LearnerDx.where(:end_dx_id => @e_id, :user_id => current_user.id).first

		@html = "<span class='glyphicon glyphicon-list-alt' style='color: green', id='" + @e.id.to_s + "'></span>"
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
	    
		@html = "<span class='glyphicon glyphicon-list-alt' style='color: gray', id='" + @e_id.to_s + "'></span>"
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
		if !@i.include?("e_")
  	    	@html = @html + "<span class='dx-toggle' id='" + @i + "'>" + n.name + "<span class='glyphicon glyphicon-menu-right'></span></span></td></tr></table>"
		else
			# Pop-up progress bar for diagnosis
		   	if !@ldx.nil?
		   	     @meter = ((@ldx.recent_correct + @ldx.recent_excellent)/(@ldx.recent_correct + @ldx.recent_excellent + @ldx.recent_incorrect))
		   	else
		   	     @meter = 0
		    	end

	      	    	@popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 50%; display: none; z-index: 30">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: ' + (@meter * 100).to_s + '%">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + n.name + ': ' + (@meter * 100).to_s + '% correct</span>
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

		@reviewhtml = " <span class='add', id='" + n.id.to_s + "'><span class='glyphicon glyphicon-list-alt' style='color: gray'></span></span> "

		if (!@ldx.nil?)
		    if @ldx.review_list == true # Override default set above
			@reviewhtml = " <span class='remove', id='" + n.id.to_s + "'><span class='glyphicon glyphicon-list-alt' style='color: green'></span></span> "
		    end
		end

  	    	@html = @html + "<span class='endDx'>" + n.name + @reviewhtml + @categorytext + "</span>" + @popup + "</td></tr></table>"	
		end
	    end

	    # Add key diagnoses if level 1
 	    if @level == "l1" and (current_user.year_of_training == "1" or current_user.year_of_training == "2")
	    	@keydx.each do |k|
		@html = @html + "<table><tr><td>"
		@ldx = LearnerDx.where(:end_dx_id => k.id).where(:user_id => current_user.id).first
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
			# Pop-up progress bar for diagnosis
		   	if !@ldx.nil?
		   	     @meter = ((@ldx.recent_correct + @ldx.recent_excellent)/(@ldx.recent_correct + @ldx.recent_excellent + @ldx.recent_incorrect))
		   	else
		   	     @meter = 0
		    	end

	      	    	@popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 50%; display: none; z-index: 30">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: ' + (@meter * 100).to_s + '%">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + k.name + ': ' + (@meter * 100).to_s + '% correct</span>
	      </div> '

		@reviewhtml = " <span class='add', id='" + k.id.to_s + "'><span class='glyphicon glyphicon-list-alt' style='color: gray'></span></span> "

		if (!@ldx.nil?)
		    if @ldx.review_list == true # Override default set above
			@reviewhtml = " <span class='remove', id='" + k.id.to_s + "'><span class='glyphicon glyphicon-list-alt' style='color: green'></span></span> "
		    end
		end

  	    	@html = @html + "<span class='endDx'>" + k.name + @reviewhtml + "<strong class='key-strong'>Key condition</strong></span>" + @popup + "</td></tr></table>"
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
	    @missed = LearnerDx.where('missed_dx > ?', 0).where(:user_id => current_user.id).limit(5).order('accuracy asc, cases_attempted desc')
	    if params[:type]
		@missed = LearnerDx.where('missed_dx > ?', 0).where(:user_id => current_user.id).limit(5).order('missed_dx desc, cases_attempted desc')
	    end
	    if params[:id]
	    	@currentreport = StudentReport.where(:id => params[:id]).first
		@expertreport = ExpertReport.where(:id => @currentreport.expert_report_id).first
		@expert_sentences = @expertreport.report_text.split(".")
		@student_sentences = @currentreport.report_text.split(".")
	    end
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

	    @studentreports = StudentReport.where(:user_id => current_user.id).order("created_at desc").limit(10)

	    if params[:id]
	    	@currentreport = StudentReport.where(:id => params[:id]).first
		@expertreport = ExpertReport.where(:id => @currentreport.expert_report_id).first
		@expert_sentences = @expertreport.report_text.split(".")
		@student_sentences = @currentreport.report_text.split(".")
	    end
	    if params[:count]
		@offset = @pageno - params[:count].to_i
		@studentreports = StudentReport.where(:user_id => current_user.id).order("created_at desc").offset(@offset*10).limit(10)

	    end

end

def skillmeters
	    @pagetype = "skillmeters"
	    @l1 = LearnerLevel1.where(:user_id => current_user.id)	
end

end
