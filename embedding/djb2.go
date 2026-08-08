package embedding

// DJB2 hash — Daniel J. Bernstein's hash function.
// Produces uniform distribution over string inputs.
// Used for deterministic workspace-to-embedding-provider mapping.
func djb2(s string) uint32 {
	h := uint32(5381)
	for i := 0; i < len(s); i++ {
		h = h*33 + uint32(s[i])
	}
	return h
}
