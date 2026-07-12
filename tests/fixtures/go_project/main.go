package main

import (
	"net/http"

	"example.com/app/handlers"
	"example.com/app/services"
)

var registry = map[string]bool{}

// init is invoked by the Go runtime — never dead.
func init() {
	registry["main"] = true
}

// main is the program entry point.
func main() {
	mux := http.NewServeMux()
	svc := &services.UserService{}
	handler := handlers.NewUserHandler(svc)
	handler.Register(mux)

	// Cross-service HTTP client call — links to the /users endpoint.
	resp, _ := http.Get("http://localhost:8080/users")
	_ = resp
}
