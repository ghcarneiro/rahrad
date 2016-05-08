class ApplicationController < ActionController::Base
  # Prevent CSRF attacks by raising an exception.
  # For APIs, you may want to use :null_session instead.
  protect_from_forgery with: :exception

  # For devise, always make sure the user is logged in first
  before_action :authenticate_user!

	before_filter :set_current_user

	def set_current_user
  		LearnerDx.current_user = current_user
	end

  before_filter :configure_permitted_parameters, if: :devise_controller?

  protected

  def configure_permitted_parameters
    devise_parameter_sanitizer.for(:sign_up) { |u| u.permit(:firstname, :lastname, :year_of_training, :email, :password, :password_confirmation, :id, :expert_report_id) }
    devise_parameter_sanitizer.for(:sign_in) { |u| u.permit(:firstname, :lastname, :year_of_training, :id, :expert_report_id) }
    devise_parameter_sanitizer.for(:account_update) { |u| u.permit(:firstname, :lastname, :year_of_training, :email, :password, :password_confirmation, :current_password, :id, :expert_report_id) }
  end

	before_filter :set_cache_headers

	  private

 	 	def set_cache_headers
  	  	response.headers["Cache-Control"] = "no-cache, no-store, max-age=0, must-revalidate"
  	  	response.headers["Pragma"] = "no-cache"
  	  	response.headers["Expires"] = "Fri, 01 Jan 1990 00:00:00 GMT"
 	 	end
end
