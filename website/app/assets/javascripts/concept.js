$(document).ready(function(){
    $.ajaxSetup({ cache: false });

    // Dropdown menu control
    $(document).on('click', '.dx-toggle', function(){
	if ($(this).children('span').hasClass('glyphicon-menu-right')) {
       	 	$(this).children('span').removeClass('glyphicon-menu-right');
       		$(this).children('span').addClass('glyphicon-menu-down');
    	}
	else {
       	 	$(this).children('span').removeClass('glyphicon-menu-down');
       		$(this).children('span').addClass('glyphicon-menu-right');
	}

	if ($(this).next().hasClass('subdata')) {
        	$(this).next().slideToggle();
	}
	else {
		var level = $(this).attr("id");
        	// Get data from model
		$.ajax({
  			type:"GET",
  			url:"data",
  			dataType:"html",
  			data: {l: level},
  			success: function(data){
    				$("#" + level).after(data);
  			}
		});
	}
    }); 

    // Control progress bars for diagnoses
    $(document).on('mouseenter mouseleave', '.endDx', function(){
	$(this).next().stop().slideToggle();
    });

    // Adding diagnoses to review list
    $(document).on('click', '.add', function(){
		var id = $(this).attr("id");
		var $this = $(this)
        	// Get data from model
		$.ajax({
  			type:"GET",
  			url:"add",
  			dataType:"html",
  			data: {c: id},
  			success: function(data) {
				$this.removeClass('add');
				$this.addClass('remove');
				$this.html(data);
			}
		});

    }); 

    // Removing diagnoses from review list
    $(document).on('click', '.remove', function(){
		var id = $(this).attr("id");
		var $this = $(this)
        	// Get data from model
		$.ajax({
  			type:"GET",
  			url:"remove",
  			dataType:"html",
  			data: {c: id},
  			success: function(data) {
				$this.removeClass('remove');
				$this.addClass('add');
				$this.html(data);
			}
		});

    }); 
});
