module DeviseHelper
   def devise_error_messages!
      return "" if resource.errors.empty?

      return resource.errors
   end

  def radio_button_selected(year)
    selected_year = params[:year_of_training] || "1"
    
    if selected_year == year
      f.radio_button :year_of_training, year, :checked => true
    else
      radio_button :year_of_training, year
    end
  end
end
