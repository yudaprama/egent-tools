package memory

import "errors"

// ErrMemoryNotFound is returned by Store.Get or Store.Delete when no
// memory entry exists for the requested (userID, key) pair. Callers
// (especially the memory tools) can use errors.Is to distinguish a
// missing-memory condition from a transport / store error.
//
// Note: the MuninnStore produces this error when neither the ID cache nor an
// activation sweep locates the requested engram.
var ErrMemoryNotFound = errors.New("memory: entry not found")
