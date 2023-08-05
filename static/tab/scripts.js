
$('.tabs a').click(function(){
	$this = $(this);
   
  
	$('.panel').hide();
	$('.tabs').removeClass('active').addClass('inactive');
	$this.addClass('active').blur();
	
	var panelContainerColor = $this.css('background-color');
  
	$('.panelContainer').css({backgroundColor: panelContainerColor});
	
	var panel = $this.attr('href');
	
	$(panel).fadeIn(350);
	
	return false;
   
  });//end click
  
   
  
  $('.tabs li:first a').click();
