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
	    @dxlevel1s = DxLevel1.all
	    @learner_level1s = LearnerLevel1.where(:user_id => current_user.id)
	    @learner_level2s = LearnerLevel2.where(:user_id => current_user.id)
	    @learner_level3s = LearnerLevel3.where(:user_id => current_user.id)
	    @learner_dxes = LearnerDx.where(:user_id => current_user.id)
            #@nodxlevel3s = DxLevel2.includes(:dx_level3s).where(:dx_level3s => {:dxable_id => nil})
	    #@hasdxlevel3s = DxLevel2.includes(:dx_level3s).where("dx_level3s.dxable_id IS NOT NULL")

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
