package models

// Reader reads bytes.
type Reader interface {
	Read() int
}

// Closer closes a resource.
type Closer interface {
	Close() error
}

// ReadCloser embeds two interfaces (interface embedding → EXTENDS heritage).
type ReadCloser interface {
	Reader
	Closer
}
