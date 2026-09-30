// Substring and byte-set search over Go strings, backed by the dispatched sz_find family.
//
// File: golang/find.go
// Author: Ash Vardanian

package sz

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
	var matchPtr C.sz_cptr_t
	check(C.sz_find_best(strPtr, C.sz_size_t(strLen), substrPtr, C.sz_size_t(substrLen), &matchPtr, capabilities(), nil))
	return matchPtr != nil
}

// Index returns the index of the first instance of `substr` in `str`, or -1 if `substr`
// is not present.
// https://pkg.go.dev/strings#Index
func Index(str string, substr string) int64 {
	substrLen := len(substr)
	if substrLen == 0 {
		return 0
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	var matchPtr C.sz_cptr_t
	check(C.sz_find_best(strPtr, C.sz_size_t(strLen), substrPtr, C.sz_size_t(substrLen), &matchPtr, capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// LastIndex returns the index of the last instance of `substr` in `str`, or -1 if `substr`
// is not present.
// https://pkg.go.dev/strings#LastIndex
func LastIndex(str string, substr string) int64 {
	substrLen := len(substr)
	strLen := int64(len(str))
	if substrLen == 0 {
		return strLen
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	var matchPtr C.sz_cptr_t
	check(C.sz_rfind_best(strPtr, C.sz_size_t(strLen), substrPtr, C.sz_size_t(substrLen), &matchPtr, capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// IndexByte returns the index of the first instance of a byte in `str`, or -1 if a byte
// is not present.
// https://pkg.go.dev/strings#IndexByte
func IndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	var matchPtr C.sz_cptr_t
	check(C.sz_find_byte_best(strPtr, C.sz_size_t(strLen), cPtr, &matchPtr, capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// LastIndexByte returns the index of the last instance of a byte in `str`, or -1 if a byte
// is not present.
// https://pkg.go.dev/strings#LastIndexByte
func LastIndexByte(str string, c byte) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	cPtr := (*C.char)(unsafe.Pointer(&c))
	var matchPtr C.sz_cptr_t
	check(C.sz_rfind_byte_best(strPtr, C.sz_size_t(strLen), cPtr, &matchPtr, capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// byteset packs the bytes of `set` into an sz_byteset_t's four words, like sz_byteset_add_u8.
func byteset(set string) (words [4]uint64) {
	for i := 0; i < len(set); i++ {
		words[set[i]>>6] |= 1 << (set[i] & 63)
	}
	return words
}

// IndexAny returns the index of the first instance of any byte from `substr` in `str`, or -1
// if none are present. Note: This is byte-set based (ASCII/bytes), not Unicode rune
// semantics like strings.IndexAny.
// https://pkg.go.dev/strings#IndexAny
func IndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	set := byteset(substr)
	var matchPtr C.sz_cptr_t
	check(C.sz_find_byteset_best(strPtr, C.sz_size_t(strLen), (*C.sz_byteset_t)(unsafe.Pointer(&set)), &matchPtr,
		capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// LastIndexAny returns the index of the last instance of any byte from `substr` in `str`, or
// -1 if none are present. Note: This is byte-set based (ASCII/bytes), not Unicode rune
// semantics like strings.LastIndexAny.
// https://pkg.go.dev/strings#LastIndexAny
func LastIndexAny(str string, substr string) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := len(str)
	set := byteset(substr)
	var matchPtr C.sz_cptr_t
	check(C.sz_rfind_byteset_best(strPtr, C.sz_size_t(strLen), (*C.sz_byteset_t)(unsafe.Pointer(&set)), &matchPtr,
		capabilities(), nil))
	if matchPtr == nil {
		return -1
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(strPtr)))
}

// Count returns the number of overlapping or non-overlapping instances of `substr` in `str`.
// If `substr` is an empty string, returns 1 + the length of the `str`.
// https://pkg.go.dev/strings#Count
func Count(str string, substr string, overlap bool) int64 {
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	strLen := int64(len(str))
	substrPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(substr)))
	substrLen := int64(len(substr))

	if substrLen == 0 {
		return 1 + strLen
	}
	if strLen == 0 || strLen < substrLen {
		return 0
	}

	step := substrLen
	if overlap {
		step = 1
	}
	count := int64(0)
	for strLen > 0 {
		var matchPtr C.sz_cptr_t
		check(C.sz_find_best(strPtr, C.sz_size_t(strLen), substrPtr, C.sz_size_t(substrLen), &matchPtr,
			capabilities(), nil))
		if matchPtr == nil {
			break
		}
		count += 1
		strLen -= step + int64(uintptr(unsafe.Pointer(matchPtr))-uintptr(unsafe.Pointer(strPtr)))
		strPtr = (*C.char)(unsafe.Add(unsafe.Pointer(matchPtr), step))
	}

	return count
}
