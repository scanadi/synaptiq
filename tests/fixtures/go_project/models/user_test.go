package models

import "testing"

// TestNewUser is a Go test function — an entry point, never dead.
func TestNewUser(t *testing.T) {
	u := NewUser("alice")
	if u.Name != "alice" {
		t.Fatalf("got %q", u.Name)
	}
	if got := helperExpected(); got != "alice" {
		t.Fatalf("helper got %q", got)
	}
}

// helperExpected is an unexported test helper — exempt because it lives in a
// _test.go file.
func helperExpected() string {
	return "alice"
}
