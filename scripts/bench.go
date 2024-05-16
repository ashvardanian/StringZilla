package main

import (
	"fmt"
	"time"
	"strings"
	sz "../go/stringzilla"
)

func main() {
    
    str := strings.Repeat("0123456789", 100000) + "something"
    pat := "some"

	fmt.Println("Contains")
    t := time.Now()
    for i := 0; i < 1; i++ {
        strings.Contains( str, pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tstrings.Contains" )

    t = time.Now()
    for i := 0; i < 1; i++ {
        sz.Contains( str,pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tsz.Contains" )

	fmt.Println("Index")
    t = time.Now()
    for i := 0; i < 1; i++ {
        strings.Index( str, pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tstrings.Index" )

    t = time.Now()
    for i := 0; i < 1; i++ {
        sz.Index( str,pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tsz.Index" )

	fmt.Println("IndexAny")
    t = time.Now()
    for i := 0; i < 1; i++ {
        strings.IndexAny( str, pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tstrings.IndexAny" )

    t = time.Now()
    for i := 0; i < 1; i++ {
        sz.IndexAny( str,pat )
    }
    fmt.Println( "  ", time.Since(t) , "\tsz.IndexAny" )


}
