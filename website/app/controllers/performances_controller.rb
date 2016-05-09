class PerformancesController < ApplicationController
	def index
	    @dxlevel1s = DxLevel1.all
	    categories = DxLevel1.all
	    @dxarray_name = []
	    @dxarray_data_total =[]
	    @dxarray_data_good =[]
	    @dxarray_data_excellent =[]

# Lazy High Charts version
	    categories.each do | category |
	    @dxarray_data_total.push({
    	        :name => category.name,
		:y => category.all_total
	    })
	    @dxarray_data_good.push(category.all_good)
	    @dxarray_data_excellent.push(category.all_excellent)
	    end

@h = LazyHighCharts::HighChart.new('graph') do |f|
f.title(:text => 'Body Systems Syllabus: Key Conditions', :align => 'center')
#f.legend(:align => 'center', :verticalAlign => 'top', :width => 400)
f.options[:chart][:type] = "line"
f.options[:chart][:polar] = "true"
f.series(:name=>'Reports attempted', :data=> @dxarray_data_total, :type => 'area', :color => '#E11313', :pointPlacement => 'on')
f.series(:name=>'Correct diagnosis', :data=> @dxarray_data_good, :type => 'area', :color => '#FFDE00', :pointPlacement => 'on')
f.series(:name=>'Excellent reports', :data=> @dxarray_data_excellent, :type => 'area', :color => '#0EE718', :pointPlacement => 'on')
f.xAxis(:type => 'category', :tickmarkPlacement => 'on', :lineWidth => 0)
f.yAxis(:labels => { :enabled => false }, :gridLineInterpolation => 'polygon', :min => 0, :max => 100)
f.tooltip(:shared => true, :pointFormat => '<span style="color:{point.color}">@</span> {series.name}: <b>{point.y:,1.f}%</b><br/>')
f.plotOptions(:series => {:fillOpacity => 1, :marker => {:enabled => false, :states => {:hover => {:enabled => false}}}})
end

	end

def concept
	    @conceptlist = "public/conceptlist_" + current_user.id.to_s
	    @dxlevel1s = DxLevel1.all
	    @learner_level1s = LearnerLevel1.where(:user_id => current_user.id)
	    @learner_level2s = LearnerLevel2.where(:user_id => current_user.id)
	    @learner_level3s = LearnerLevel3.where(:user_id => current_user.id)
	    @learner_dxes = LearnerDx.where(:user_id => current_user.id)
            #@nodxlevel3s = DxLevel2.includes(:dx_level3s).where(:dx_level3s => {:dxable_id => nil})
	    #@hasdxlevel3s = DxLevel2.includes(:dx_level3s).where("dx_level3s.dxable_id IS NOT NULL")

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
  	    	@html = @html + n.name + "<span class='glyphicon glyphicon-menu-right' id='" + @i + "'></span></td></tr></table>"
		else
	      @popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 200px; display: none">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: 50px">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + n.name + ': X% correct</span>
	      </div>'
  	    	@html = @html + "<span class='endDx'>" + n.name + "</span>" + @popup + "</td></tr></table>"	
		end
	    end

	    # Add key diagnoses if level 1
	    if @level == "l1" and current_user.year_of_training == "1"
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
	      @popup = '<div style="width: 250px; height: 50px; background-color: white; border: 1px solid #CCCCCC; position: absolute; left: 200px; display: none">
	        <div class="progress" style="width: 200px; position: relative; left: 20px; top: 10px">
	          <div class="progress-bar" style="width: 50px">
	          </div>
	        </div>
	        <span style="font-size: 10px; position: relative; left: 10px; top: -10px">' + k.name + ': X% correct</span>
	      </div>'
  	    	@html = @html + "<span class='endDx'>" + k.name +  "</span>" + @popup + "</td></tr></table>"
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
	    @level = 1
	    @learner_level1s = LearnerLevel1.where(:user_id => current_user.id)	
	    # Display the appropriate level 2 data for the selected level 1 diagnosis  
	    if params[:l1]  
		@level = 2
		# Get level 2 diagnoses
		@learner1 = LearnerLevel1.where(:id => params[:l1]).first
		@dxlevel1 = DxLevel1.where(:id => @learner1.dx_level1_id).first
		@dxlevel2s = DxLevel2.where(:dx_level1_id => @dxlevel1.id)

		# Get key conditions
		@keyconditions = EndDx.where(:dxable_type => "DxLevel1").where(:dxable_id => @dxlevel1.id)
	    end
	    if params[:l2]  
		@level = 3
		# Get level 3 diagnoses
		@learner2 = LearnerLevel2.where(:id => params[:l2]).first
		@dxlevel2 = DxLevel2.where(:id => @learner2.dx_level2_id).first
		@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
		@dxlevel3s = DxLevel3.where(:dx_level2_id => @dxlevel2.id)
		if @dxlevel3s.nil?
		    @level = 4
		    @end_dxes = EndDx.where(:dxable_id => @dxlevel2.id, :dxable_type => "DxLevel2")
		end
	    end
	    if params[:l3]  
		@level = 4
		# Get level 3 diagnoses
		@learner3 = LearnerLevel3.where(:id => params[:l3]).first
		@dxlevel3 = DxLevel3.where(:id => @learner3.dx_level3_id).first
		@dxlevel2 = DxLevel2.where(:id => @dxlevel3.dx_level2_id).first
		@dxlevel1 = DxLevel1.where(:id => @dxlevel2.dx_level1_id).first
		@end_dxes = EndDx.where(:dxable_id => @dxlevel3.id, :dxable_type => "DxLevel3")
	    end
end

def report_hx
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

end
