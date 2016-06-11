class LearnerModelsController < ApplicationController
  def create
    @learner_model = LearnerModel.new(params[:learner_model]) 
    @learner_model.save
    redirect_to @learner_model
  end

  private
    def learner_model_params
      params.require(:learner_model).permit(:abdominal, :cardiothoracic)
    end
end
