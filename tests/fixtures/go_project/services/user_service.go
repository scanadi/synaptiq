package services

import "example.com/app/models"

// UserService loads users.
type UserService struct{}

// FindUser resolves a user by name; delegates to an unexported helper.
func (s *UserService) FindUser(name string) *models.User {
	return s.build(name)
}

// build is unexported but reached via s.build — must not be flagged dead.
func (s *UserService) build(name string) *models.User {
	return models.NewUser(name)
}
