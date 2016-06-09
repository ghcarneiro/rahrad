$(document).ready(function(){
    $(".sentence-toggle").click(function(){
	if ($(this).children('span').hasClass('glyphicon-menu-right')) {
       	 	$(this).children(':first-child').next().removeClass('glyphicon-menu-right');
       		$(this).children(':first-child').next().addClass('glyphicon-menu-down');
		$(this).children(':first-child').text('Hide ');
    	}
	else {
       	 	$(this).children(':first-child').next().removeClass('glyphicon-menu-down');
       		$(this).children(':first-child').next().addClass('glyphicon-menu-right');
		$(this).children(':first-child').text('View ');
	}

    $(this).next().slideToggle();

    }); 

    $(document).on('click', '.add-review', function(){
		var id = $(this).attr("id");
		var $this = $(this)
        	// Get data from model
		$.ajax({
  			type:"GET",
  			url:"add",
  			dataType:"html",
  			data: {l: id},
  			success: function(data) {
				$this.removeClass('add-review');
				$this.addClass('remove-review');
				$this.html(data);
			}
		});

    }); 
    $(document).on('click', '.remove-review', function(){
		var id = $(this).attr("id");
		var $this = $(this)
        	// Get data from model
		$.ajax({
  			type:"GET",
  			url:"remove",
  			dataType:"html",
  			data: {l: id},
  			success: function(data) {
				$this.removeClass('remove-review');
				$this.addClass('add-review');
				$this.html(data);
			}
		});

    }); 



});
