require_relative "application_controller"
require_relative "../services/user_service"

class UsersController < ApplicationController
  def show
    authenticate
    service = UserService.new
    user = service.find_user("alice")
    user.display
  end
end
