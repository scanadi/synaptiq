require_relative "../../lib/user"

class UserService
  def find_user(name)
    User.new(name)
  end
end
