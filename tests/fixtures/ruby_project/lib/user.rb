require_relative "greeter"

class User
  include Greeter

  attr_reader :name

  def initialize(name)
    @name = name
  end

  def display
    greet(@name)
  end
end
