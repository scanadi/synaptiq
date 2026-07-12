package models

// UserID is a type alias (→ TypeAlias node).
type UserID = int

// User is an application user; it embeds Base.
type User struct {
	Name string
	Base
}

// NewUser constructs a User (exercised across packages).
func NewUser(name string) *User {
	return &User{Name: name}
}

// Display renders the user.
func (u *User) Display() string {
	return u.Name
}

// unusedHelper is never referenced anywhere — dead code.
func unusedHelper() string {
	return "this function is never called"
}
