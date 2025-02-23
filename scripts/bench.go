package main

import (
	"fmt"
	"strings"
	"time"

	sz "../go/stringzilla"
)

func main() {

	str := strings.Repeat("0123456789", 10000) + "something"
	pat := "some"

	fmt.Println("Contains")
	t := time.Now()
	for i := 0; i < 1; i++ {
		strings.Contains(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tstrings.Contains")

	t = time.Now()
	for i := 0; i < 1; i++ {
		sz.Contains(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tsz.Contains")

	fmt.Println("Index")
	t = time.Now()
	for i := 0; i < 1; i++ {
		strings.Index(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tstrings.Index")

	t = time.Now()
	for i := 0; i < 1; i++ {
		sz.Index(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tsz.Index")

	fmt.Println("IndexAny")
	t = time.Now()
	for i := 0; i < 1; i++ {
		strings.IndexAny(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tstrings.IndexAny")

	t = time.Now()
	for i := 0; i < 1; i++ {
		sz.IndexAny(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tsz.IndexAny")

	str = strings.Repeat("0123456789", 100000) + "something"
	pat = "123456789"
	fmt.Println("Count")
	t = time.Now()
	for i := 0; i < 1; i++ {
		strings.Count(str, pat)
	}
	fmt.Println("  ", time.Since(t), "\tstrings.Count")

	t = time.Now()
	for i := 0; i < 1; i++ {
		sz.Count(str, pat, false)
	}
	fmt.Println("  ", time.Since(t), "\tsz.Count")

}
