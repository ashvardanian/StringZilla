// StringZilla is a SIMD-accelerated string library modern CPUs, written in C 99,
// and using AVX2, AVX512, Arm NEON, and SVE intrinsics to accelerate processing.
//
// The GoLang binding is intended to provide a simple interface to a precompiled
// shared library, available on GitHub: https://github.com/ashvardanian/stringzilla
//
// It requires Go 1.24 or newer to leverage the `cGo` `noescape` and `nocallback`
// directives. Without those the latency of calling C functions from Go is too high
// to be useful for string processing.
package sz

// #cgo CFLAGS: -O3
// #cgo LDFLAGS: -lstringzilla_shared
// #cgo noescape sz_find
// #cgo noescape sz_find_byte
// #cgo noescape sz_rfind
// #cgo noescape sz_rfind_byte
// #cgo noescape sz_find_char_from
// #cgo noescape sz_rfind_char_from
// #cgo noescape sz_look_up_transform
// #cgo noescape sz_hamming_distance
// #cgo noescape sz_hamming_distance_utf8
// #cgo noescape sz_edit_distance
// #cgo noescape sz_edit_distance_utf8
// #cgo noescape sz_alignment_score
// #cgo nocallback sz_find
// #cgo nocallback sz_find_byte
// #cgo nocallback sz_rfind
// #cgo nocallback sz_rfind_byte
// #cgo nocallback sz_find_char_from
// #cgo nocallback sz_rfind_char_from
// #cgo nocallback sz_look_up_transform
// #cgo nocallback sz_hamming_distance
// #cgo nocallback sz_hamming_distance_utf8
// #cgo nocallback sz_edit_distance
// #cgo nocallback sz_edit_distance_utf8
// #cgo nocallback sz_alignment_score
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
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
	matchPtr := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(matchPtr) - uintptr(unsafe.Pointer(strPtr)))
}

// Index returns the index of the last instance of `substr` in `str`, or -1 if `substr` is not present.
// https://pkg.go.dev/strings#LastIndex
func LastIndex(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)
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
// If `substr` is an empty string, returns 1 + the number of Unicode code points in `str`.
// https://pkg.go.dev/strings#Count
func Count(str string, substr string, overlap bool) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := len(substr)

	if substrLen == 0 {
		return 1 + len([]rune(str))
	}
	if strLen == 0 || strLen < substrLen {
		return 0
	}

	count := int64(0)
	if overlap == true {
		for strLen > 0 {
			ret := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if ret == nil {
				break
			}
			count += 1
			strLen -= (1 + int64(uintptr(ret)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(ret, 1))
		}
	} else {
		for strLen > 0 {
			ret := unsafe.Pointer(C.sz_find(strPtr, C.ulong(strLen), substrPtr, C.ulong(substrLen)))
			if ret == nil {
				break
			}
			count += 1
			strLen -= (substrLen + int64(uintptr(ret)-uintptr(unsafe.Pointer(strPtr))))
			strPtr = (*C.char)(unsafe.Add(ret, substrLen))
		}
	}

	return count
}
