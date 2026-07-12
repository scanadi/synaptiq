package models

// Base is embedded into User (struct embedding → EXTENDS heritage).
type Base struct {
	ID int
}

// Describe is promoted onto the embedding struct.
func (b *Base) Describe() string {
	return "base"
}
