package sz

// #cgo CFLAGS: -g -mavx2
// #include <stdlib.h>
// #include <../../include/stringzilla/stringzilla.h>
import "C"

// -Wall -O3

import (
	"unsafe"
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

func LastIndex( str string, pat string ) int64 {
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
func RFind( str string, pat string ) int64 {
    return LastIndex(str,pat)
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

func Count( str string, pat string, overlap bool ) int64 {
    cstr := (*C.char)(unsafe.Pointer(unsafe.StringData(str)))
    cpat := (*C.char)(unsafe.Pointer(unsafe.StringData(pat)))
    strlen := int64(len(str))
    patlen := int64(len(pat))

    if strlen == 0 || patlen == 0 || strlen < patlen {
        return 0
    }

    count := int64(0);
    if overlap == true {
        for strlen > 0 {
            ret := unsafe.Pointer(C.sz_find( cstr, C.ulong(strlen), cpat, C.ulong(patlen) ))
            if ret == nil {
                break
            }
            count += 1
            strlen -= ( 1 + int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr))) )
            cstr = (*C.char)(unsafe.Add(ret,1))
        }
    } else {
        for strlen > 0 {
            ret := unsafe.Pointer(C.sz_find( cstr, C.ulong(strlen), cpat, C.ulong(patlen) ))
            if ret == nil {
                break
            }
            count += 1
            strlen -= (patlen+int64(uintptr(ret)-uintptr(unsafe.Pointer(cstr))))
            cstr = (*C.char)(unsafe.Add(ret,patlen))
        }
    }

    return count

}



