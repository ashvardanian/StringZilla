package main

import (
	"fmt"
	"strings"
	"runtime"
	sz "../go/stringzilla"
)

func assertEqual[T comparable](act T, exp T) int {
    if exp == act {
        return 0
    }
	_, _, line, _ := runtime.Caller(1)
    fmt.Println("")
    fmt.Println("  ERROR line ",line," expected (",exp,") is not equal to actual (",act,")")
	return 1
}

func main() {
    
    str := strings.Repeat("0123456789", 100000) + "something"
    pat := "some"
	ret := 0

	fmt.Print("Contains ... ")
    ret |= assertEqual( sz.Contains( "", "" ), true )
    ret |= assertEqual( sz.Contains( "test", "" ), true )
    ret |= assertEqual( sz.Contains( "test", "s" ), true )
    ret |= assertEqual( sz.Contains( "test", "test" ), true )
    ret |= assertEqual( sz.Contains( "test", "zest" ), false )
    ret |= assertEqual( sz.Contains( "test", "z" ), false )
	if ( ret == 0 ) {
		fmt.Println("successful")
	}

	fmt.Print("Index ... ")
    assertEqual( strings.Index( str, pat ), int(sz.Index( str,pat )) )
    assertEqual( sz.Index( "","" ), 0 )
    assertEqual( sz.Index( "test","" ), 0 )
    assertEqual( sz.Index( "test","t" ), 0 )
    assertEqual( sz.Index( "test","s" ), 2 )
	fmt.Println("successful")

	fmt.Print("IndexAny ... ")
    assertEqual( strings.IndexAny( str, pat ), int(sz.IndexAny( str,pat )) )
    assertEqual( sz.IndexAny( "test", "st" ), 0 )
    assertEqual( sz.IndexAny( "west east", "ta" ), 3)
	fmt.Println("successful")

	fmt.Print("Count ... ")
    //assertEqual( strings.Count( str, pat ), int(sz.Count( str,pat,false )) )
    assertEqual( sz.Count( "aaaaa", "a", false ),  5 )
    assertEqual( sz.Count( "aaaaa", "aa", false ), 2 )
    assertEqual( sz.Count( "aaaaa", "aa", true ),  4 )
	fmt.Println("successful")


}
