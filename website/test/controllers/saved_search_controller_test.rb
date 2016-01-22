require 'test_helper'

class SavedSearchControllerTest < ActionController::TestCase
  test "should get index" do
    get :index
    assert_response :success
  end

  test "should get edit_repeat_search" do
    get :edit_repeat_search
    assert_response :success
  end

end
