package sz

// #cgo CFLAGS: -g -mavx2
// #include <stdlib.h>
// #include <../../include/stringzilla/stringzilla.h>
import "C"

// -Wall -O3

import (
	//"fmt"
	//"time"
	"unsafe"
	//"strings"
)

/*
// Passing a C function pointer around in go isn't working
//type searchFunc func(*C.char, C.ulong, *C.char, C.ulong)C.sz_cptr_t
//func _search( str string, pat string, searchFunc func(*C.char, C.ulong, *C.char, C.ulong)C.sz_cptr_t) uintptr {
func _search( str string, pat string, searchFunc C.sz_find_t ) uintptr {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := len(str)
    patlen := len(pat)
    ret := unsafe.Pointer( searchFunc(cstr, C.ulong(strlen), cpat, C.ulong(patlen)) )
    return ret
}
*/

func Contains( str string, pat string ) bool {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := len(str)
    patlen := len(pat)
    ret := unsafe.Pointer(C.sz_find( cstr, C.ulong(strlen), cpat, C.ulong(patlen) )) 
	//ret := _search( str, pat, C.sz_find_t(C.sz_find) )
    return ret != nil
}

func Index( str string, pat string ) int64 {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := len(str)
    patlen := len(pat)
    ret := unsafe.Pointer(C.sz_find( cstr, C.ulong(strlen), cpat, C.ulong(patlen) )) 
	if ret == nil  {
		return 0
	}
    return int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr)))
}

func Find( str string, pat string ) int64 {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := len(str)
    patlen := len(pat)
    ret := unsafe.Pointer(C.sz_find( cstr, C.ulong(strlen), cpat, C.ulong(patlen) )) 
	if ret == nil {
		return -1
	}
    return int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr)))
}

func RFind( str string, pat string ) int64 {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := len(str)
    patlen := len(pat)
    ret := unsafe.Pointer(C.sz_rfind( cstr, C.ulong(strlen), cpat, C.ulong(patlen) )) 
	if ret == nil {
		return -1
	}
    return int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr)))
}

func IndexAny( str string, charset string ) int64 {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(charset)))
    strlen := len(str)
    patlen := len(charset)
    ret := unsafe.Pointer(C.sz_find_char_from( cstr, C.ulong(strlen), cpat, C.ulong(patlen) )) 
	if ret == nil {
		return -1
	}
    return int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr)))
}
func FindCharFrom( str string, charset string ) int64 {
	return IndexAny( str, charset )
}



/*
func Contains( s *C.char, s_length int, pattern *C.char, pattern_length int ) bool {
    c := unsafe.Pointer(C.sz_find( s, C.ulong(s_length), pattern, C.ulong(pattern_length) )) 
    return c != nil
}
func Contains_sz( a C.sz_string_view_t, b C.sz_string_view_t ) bool {
    c := unsafe.Pointer(C.sz_find( a.start, a.length, b.start, b.length) ) 
    return c != nil
}
func Index( s *C.char, s_length int, pattern *C.char, pattern_length int ) uintptr {
    c := unsafe.Pointer(C.sz_find( s, C.ulong(s_length), pattern, C.ulong(pattern_length) )) 
    return uintptr(c)-uintptr(unsafe.Pointer(s)) 
}
func main() {
    
    str := strings.Repeat("0123456789", 100000) + "something"
    pat := "some"
    t := time.Now()
    for i := 0; i < 1; i++ {
        strings.Contains( str, pat )
        //strings.Index( str, pat )
    }
    fmt.Println( time.Since(t) )

    //a := C.CString(str)
    a := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    b := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    alen := len(str)
    blen := len(pat)
    sva := C.sz_string_view_t {a,(C.ulong)(alen)}
    svb := C.sz_string_view_t {b,(C.ulong)(blen)}
    fmt.Println(sva.length)
    t = time.Now()
    for i := 0; i < 1; i++ {
        Contains( a, alen, b, blen ) 
        //index( a, alen, b, blen ) 
    }
    fmt.Println( time.Since(t) )

    t = time.Now()
    for i := 0; i < 1; i++ {
        ContainsString( str,pat )
        //Contains_sz( sva, svb )
        //index( a, alen, b, blen ) 
    }
    fmt.Println( time.Since(t) )

    fmt.Println(strings.Contains( str, pat ))
    fmt.Println(Contains_sz( sva, svb ))
    fmt.Println(strings.Index( str, pat ))
    fmt.Println(Index(a,alen,b,blen))
    //fmt.Println(strings.Contains("something", "some"))
    //fmt.Println( contains( a, 9, b, 4 ) )
    //fmt.Println( len(a) )
    //fmt.Println( len(b) )
    //c := unsafe.Pointer(C.sz_find( a, 9, b, 4 )  )
    //d := *C.uchar(c)
    //fmt.Println( a )
    //fmt.Println( c )
    //fmt.Println( uintptr(c)-uintptr(unsafe.Pointer(a)) )
    //fmt.Println( C.sz_find( a, 9, b, 4 )  )
        //sz_cptr_t result = sz_find(haystack.start, haystack.length, needle.start, needle.length);

        // In JavaScript, if `indexOf` is unable to indexOf the specified value, then it should return -1
        //if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        //else { napi_create_bigint_uint64(env, result - haystack.start, &js_result); }
}
*/
