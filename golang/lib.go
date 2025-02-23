// StringZilla is a SIMD-accelerated string library modern CPUs, written in C 99,
// and using AVX2, AVX512, Arm NEON, and SVE intrinsics to accelerate processing.
//
// The GoLang binding is intended to provide a simple interface to a precompiled
// shared library, available on GitHub: https://github.com/ashvardanian/stringzilla
//
// It requires Go 1.24 or newer to leverage the `cGo` `noescape` and `nocallback`
// directives. Without those the latency of calling C functions from Go is too high
// to be useful for string processing.
//
// Unlike the native Go `strings` package, StringZilla primarily targets byte-level
// binary data processing, with less emphasis on UTF-8 and locale-specific tasks.
package sz

// #cgo CFLAGS: -O3
// #cgo LDFLAGS: -L. -L/usr/local/lib -lstringzilla_shared
// #cgo noescape sz_find
// #cgo noescape sz_find_byte
// #cgo noescape sz_rfind
// #cgo noescape sz_rfind_byte
// #cgo noescape sz_find_char_from
// #cgo noescape sz_rfind_char_from
// #cgo nocallback sz_find
// #cgo nocallback sz_find_byte
// #cgo nocallback sz_rfind
// #cgo nocallback sz_rfind_byte
// #cgo nocallback sz_find_char_from
// #cgo nocallback sz_rfind_char_from
// #define SZ_DYNAMIC_DISPATCH 1
// #include <stringzilla/stringzilla.h>
import "C"
import "unsafe"

// Contains reports whether `substr` is within `str`.
// https://pkg.go.dev/strings#Contains
func Contains(str string, substr string) bool {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	return matchPtr != nil
}

// Index returns the index of the first instance of `substr` in `str`, or -1 if `substr` is not present.
// https://pkg.go.dev/strings#Index
func Index(str string, substr string) int64 {
	substrLen := len(substr)
	if substrLen == 0 {
		return 0
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of `substr` in `str`, or -1 if `substr` is not present.
// https://pkg.go.dev/strings#LastIndex
func LastIndex(str string, substr string) int64 {
	substrLen := len(substr)
	strLen := int64(len(str))
	if substrLen == 0 {
		return strLen
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	matchPtr := unsafe.Pointer(C.sz_rfind(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the first instance of a byte in `str`, or -1 if a byte is not present.
// https://pkg.go.dev/strings#IndexByte
func IndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	matchPtr := unsafe.Pointer(C.sz_find_byte(strPtr, C.ulong(strLen), cPtr))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of a byte in `str`, or -1 if a byte is not present.
// https://pkg.go.dev/strings#LastIndexByte
func LastIndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	matchPtr := unsafe.Pointer(C.sz_rfind_byte(strPtr, C.ulong(strLen), cPtr))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the first instance of any byte from `substr` in `str`, or -1 if none are present.
// https://pkg.go.dev/strings#IndexAny
func IndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_find_char_from(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of any byte from `substr` in `str`, or -1 if none are present.
// https://pkg.go.dev/strings#LastIndexAny
func LastIndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_rfind_char_from(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Count returns the number of overlapping or non-overlapping instances of `substr` in `str`.
// If `substr` is an empty string, returns 1 + the length of the `str`.
// https://pkg.go.dev/strings#Count
func Count(str string, substr string, overlap bool) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := int64(len(str))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := int64(len(substr))

	if strLen == 0 || strLen < substrLen {
		return 0
	}
	if substrLen == 0 {
		return 1 + strLen
	}

	count := int64(0)
	if overlap == true {
		for strLen > 0 {
			matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if matchPtr == nil {
				break
			}
			count += 1
			strLen -= (1 + int64(uintptr(matchPtr)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(matchPtr, 1))
		}
	} else {
		for strLen > 0 {
			matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if matchPtr == nil {
				break
			}
			count += 1
			strLen -= (substrLen + int64(uintptr(matchPtr)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(matchPtr, substrLen))
		}
	}

	return count
}
