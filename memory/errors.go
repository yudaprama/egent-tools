package memory

import "errors"

// ErrMemoryNotFound is returned by Store.Delete when no memory entry
// exists for the requested (userID, key) pair. (Store.Get returns
// (nil, nil) instead — see the interface contract.)
var ErrMemoryNotFound = errors.New("memory: entry not found")
