require 'test_helper'

class TableControllerTest < ActionController::TestCase
  test "should get index" do
    get :index
    assert_response :success
  end

  test "should get display_report" do
    get :display_report
    assert_response :success
  end

end
