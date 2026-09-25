// The SIMD capabilities the dispatched StringZilla build detected on this CPU.
//
// File: golang/capabilities.go
// Author: Ash Vardanian

package sz

// #include <stringzilla/stringzilla.h>
import "C"

// Capabilities returns a string describing the detected CPU features.
// This can be used for debugging to understand which SIMD backend is being used.
func Capabilities() string {
	caps := C.sz_capabilities()
	capsStr := C.sz_capabilities_to_string(caps)
	return C.GoString(capsStr)
}
