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


# FusionCharts code
=begin 
	    categories.each do | category |
	    @dxarray_name.push({
    	        :label => category.name,
	    })
	    @dxarray_data_total.push({
    	        :value => category.all_total,
	    })
	    @dxarray_data_good.push({
    	        :value => category.all_good,
	    })
	    @dxarray_data_excellent.push({
    	        :value => category.all_excellent,
	    })
	    end

	    @key_dxarray_name = []
	    @key_dxarray_data_total =[]
	    @key_dxarray_data_good =[]
	    @key_dxarray_data_excellent =[]
	    categories.each do | category |
	    @key_dxarray_name.push({
    	        :label => category.name,
	    })
	    @key_dxarray_data_total.push({
    	        :value => category.key_total,
	    })
	    @key_dxarray_data_good.push({
    	        :value => category.key_good,
	    })
	    @key_dxarray_data_excellent.push({
    	        :value => category.key_excellent,
	    })

	    end

@chart = Fusioncharts::Chart.new({
    :height => 350,
    :width => 400,
    :type => 'radar',
    :renderAt => 'chart-container',
    :anchorAlpha => 0,
    :dataSource => {
        :chart => {
            :caption => 'Body Systems',
            :numberPrefix => '',
            :theme => 'fint',
	    :radarfillcolor => '#ffffff'
        },
	:categories => {
		:category => @dxarray_name
	},
	:dataset => [{
		:seriesname => 'Attempted reports',
		:data => @dxarray_data_total
		}, {
		:seriesname => 'Correct diagnoses',
		:data => @dxarray_data_good
		}, {
		:seriesname => 'Excellent reports',
		:data => @dxarray_data_excellent
        }]
    }
});
=end

	end


end
