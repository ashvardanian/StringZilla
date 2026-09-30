// UTF-8 counting, normalization and case-insensitive search over the dispatched sz_utf8 kernels.
//
// File: golang/utf8.go
// Author: Ash Vardanian

package sz

// #include <stdlib.h>
// #include <stringzilla/stringzilla.h>
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

var ErrInvalidUTF8 = errors.New("invalid UTF-8")

func isValidUTF8String(s string) bool {
	if len(s) == 0 {
		return true
	}
	return C.sz_utf8_find_malformed((*C.char)(unsafe.Pointer(unsafe.StringData(s))), C.sz_size_t(len(s))) == nil
}

// Utf8CaseFold applies full Unicode case folding to a UTF-8 string.
// It can expand the output, as "ß" → "ss" does, so it allocates up to 3× the input byte size.
func Utf8CaseFold(str string, validate bool) (string, error) {
	if len(str) == 0 {
		return "", nil
	}
	if validate && !isValidUTF8String(str) {
		return "", ErrInvalidUTF8
	}

	srcPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	srcLen := C.sz_size_t(len(str))
	dst := make([]byte, len(str)*3)
	var outLen C.sz_size_t
	if err := statusError(C.sz_utf8_uncased_fold_best(srcPtr, srcLen, (*C.char)(unsafe.Pointer(&dst[0])), &outLen,
		capabilities(), nil)); err != nil {
		return "", err
	}
	return string(dst[:outLen]), nil
}

// Utf8Count returns the number of Unicode codepoints in a UTF-8 string, SIMD-accelerated.
// It counts non-continuation bytes, the ones not matching the 10xxxxxx pattern, which agrees with
// utf8.RuneCount on well-formed UTF-8 but can differ on malformed input: for example, a run of
// lone continuation bytes counts as zero here, while utf8.RuneCount counts one per byte.
func Utf8Count(str string) int {
	if len(str) == 0 {
		return 0
	}
	strPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	var count C.sz_size_t
	check(C.sz_utf8_count_best(strPtr, C.sz_size_t(len(str)), &count, capabilities(), nil))
	return int(count)
}

// NormalForm selects a Unicode normalization form for Utf8Normalize.
type NormalForm int

const (
	// NFD is canonical decomposition.
	NFD NormalForm = C.sz_normal_form_nfd_k
	// NFC is canonical decomposition followed by canonical composition.
	NFC NormalForm = C.sz_normal_form_nfc_k
	// NFKD is compatibility decomposition.
	NFKD NormalForm = C.sz_normal_form_nfkd_k
	// NFKC is compatibility decomposition followed by canonical composition.
	NFKC NormalForm = C.sz_normal_form_nfkc_k
)

// Utf8Normalize returns the string normalized to the requested Unicode form.
// Malformed bytes pass through unchanged. The destination is sized for the worst-case
// compatibility decomposition (18x); composing forms never exceed that bound.
func Utf8Normalize(str string, form NormalForm) string {
	if len(str) == 0 {
		return ""
	}
	srcPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
	dst := make([]byte, len(str)*18)
	var outLen C.sz_size_t
	check(C.sz_utf8_norm_best(srcPtr, C.sz_size_t(len(str)), C.sz_normal_form_t(form),
		(*C.char)(unsafe.Pointer(&dst[0])), &outLen, capabilities(), nil))
	return string(dst[:outLen])
}

// Utf8CaseInsensitiveFind finds the first case-insensitive occurrence of `needle` in `haystack`
// using full Unicode case folding and returns byte offsets.
func Utf8CaseInsensitiveFind(haystack, needle string, validate bool) (index int64, length int64, err error) {
	if validate {
		if !isValidUTF8String(haystack) || !isValidUTF8String(needle) {
			return -1, 0, ErrInvalidUTF8
		}
	}

	// The prepared needle points into `needle`, and cgo only passes Go pointers that are pinned.
	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(unsafe.StringData(needle))
	var prepared C.sz_utf8_uncased_needle_t
	if err := statusError(C.sz_utf8_uncased_needle_init_best((*C.char)(unsafe.Pointer(unsafe.StringData(needle))),
		C.sz_size_t(len(needle)), &prepared, capabilities(), nil)); err != nil {
		return -1, 0, err
	}
	return utf8UncasedSearch(haystack, &prepared)
}

// Utf8CaseInsensitiveNeedle is a needle prepared once for repeated case-insensitive UTF-8 searches.
// It is safe for concurrent use, as searches only read it.
type Utf8CaseInsensitiveNeedle struct {
	prepared C.sz_utf8_uncased_needle_t
}

// NewUtf8CaseInsensitiveNeedle prepares a reusable case-insensitive needle over its own copy of it.
// If validate is true, the needle is validated as UTF-8.
func NewUtf8CaseInsensitiveNeedle(needle string, validate bool) (*Utf8CaseInsensitiveNeedle, error) {
	if validate && !isValidUTF8String(needle) {
		return nil, ErrInvalidUTF8
	}
	// The prepared needle points at its bytes, so they live in C memory, which cgo never checks.
	bytes := C.CString(needle)
	n := &Utf8CaseInsensitiveNeedle{}
	if err := statusError(C.sz_utf8_uncased_needle_init_best(bytes, C.sz_size_t(len(needle)), &n.prepared,
		capabilities(), nil)); err != nil {
		C.free(unsafe.Pointer(bytes))
		return nil, err
	}
	runtime.AddCleanup(n, func(bytes *C.char) { C.free(unsafe.Pointer(bytes)) }, bytes)
	return n, nil
}

// FindIn searches for the prepared needle in haystack and returns byte offsets.
func (n *Utf8CaseInsensitiveNeedle) FindIn(haystack string, validate bool) (index int64, length int64, err error) {
	if n == nil {
		return -1, 0, errors.New("nil Utf8CaseInsensitiveNeedle")
	}
	if validate && !isValidUTF8String(haystack) {
		return -1, 0, ErrInvalidUTF8
	}
	return utf8UncasedSearch(haystack, &n.prepared)
}

// utf8UncasedSearch finds a prepared needle in haystack, returning byte offsets like FindIn.
func utf8UncasedSearch(haystack string, needle *C.sz_utf8_uncased_needle_t) (index int64, length int64, err error) {
	// An empty haystack may have no data pointer to report the empty needle's match against.
	if needle.length == 0 {
		return 0, 0, nil
	}
	hPtr := (*C.char)(unsafe.Pointer(unsafe.StringData(haystack)))
	var matchPtr C.sz_cptr_t
	var matchedLen C.sz_size_t
	if err := statusError(C.sz_utf8_uncased_search_best(hPtr, C.sz_size_t(len(haystack)), needle, &matchPtr,
		&matchedLen, capabilities(), nil)); err != nil {
		return -1, 0, err
	}
	if matchPtr == nil {
		return -1, 0, nil
	}
	return int64(uintptr(unsafe.Pointer(matchPtr)) - uintptr(unsafe.Pointer(hPtr))), int64(matchedLen), nil
}
