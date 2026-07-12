package handlers

import (
	"net/http"

	"example.com/app/services"
)

// Handler is the app's HTTP handler contract (an interface node).
type Handler interface {
	Register(mux *http.ServeMux)
}

// UserHandler serves user endpoints.
type UserHandler struct {
	svc *services.UserService
}

// NewUserHandler constructs a UserHandler.
func NewUserHandler(svc *services.UserService) *UserHandler {
	return &UserHandler{svc: svc}
}

// Register wires the routes (REST endpoint definition).
func (h *UserHandler) Register(mux *http.ServeMux) {
	mux.HandleFunc("/users", h.ServeUsers)
}

// ServeUsers handles GET /users.
func (h *UserHandler) ServeUsers(w http.ResponseWriter, r *http.Request) {
	user := h.svc.FindUser("alice")
	_, _ = w.Write([]byte(user.Display()))
}
