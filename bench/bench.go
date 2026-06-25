package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	sz "github.com/ashvardanian/stringzilla/golang"
)

var sink any //? Global sink to defeat dead-code elimination

// Repeats a certain function `f` multiple times and prints the benchmark results.
func runBenchmark[T any](name string, f func() T) {
	benchResult := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink = f()
		}
	})
	fmt.Printf("%-30s: %s\n", name, benchResult.String())
}

func main() {

	// Define command-line flags.
	inputPath := flag.String("input", "", "Path to input file for benchmarking. (Required)")
	seedInt := flag.Int64("seed", 0, "Seed for the random number generator. If 0, the current time is used.")
	splitMode := flag.String("split", "tokens", "How to split input file: 'tokens' (default) or 'lines'.")
	flag.Parse()

	// Ensure input file is provided.
	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "Error: input file must be specified using the -input flag.")
		flag.Usage()
		os.Exit(1)
	}

	// Read input data from file.
	bytes, err := os.ReadFile(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input file: %v\n", err)
		os.Exit(1)
	}
	data := string(bytes)
	fmt.Printf("Benchmarking on `%s` with seed %d.\n", *inputPath, *seedInt)
	fmt.Printf("Total input length: %d\n", len(data))

	// Split the data into items based on the chosen mode.
	var items []string
	switch *splitMode {
	case "lines":
		rawLines := strings.Split(data, "\n")
		// Filter out empty lines.
		for _, line := range rawLines {
			if line != "" {
				items = append(items, line)
			}
		}
		if len(items) == 0 {
			items = []string{"default"}
		}
		// Print line statistics.
		totalLen := 0
		for _, line := range items {
			totalLen += len(line)
		}
		fmt.Printf("Total lines: %d\n", len(items))
		fmt.Printf("Average line length: %.2f\n", float64(totalLen)/float64(len(items)))
	default: // "tokens" or any other value defaults to token mode.
		items = strings.Fields(data)
		if len(items) == 0 {
			items = []string{"default"}
		}
		fmt.Printf("Total tokens: %d\n", len(items))
		fmt.Printf("Average token length: %.2f\n", float64(len(data))/float64(len(items)))
	}

	// In Go, a string is represented as a (length, data) pair. If you pass a string around,
	// Go will copy the length and the pointer but not the data pointed to.
	// It's problematic for our benchmark as it makes substring operations meaningless -
	// just comparing if a pointer falls in the range.
	// To avoid that, let's copy strings to `[]byte` and back to force a new allocation.
	for i, item := range items {
		items[i] = string([]byte(item))
	}

	// Create a seeded reproducible random number generator.
	if *seedInt == 0 {
		*seedInt = time.Now().UnixNano()
	}
	generator := rand.New(rand.NewSource(*seedInt))
	randomItem := func() string {
		return items[generator.Intn(len(items))]
	}

	fmt.Println("Running benchmark using `testing.Benchmark`.")

	runBenchmark("strings.Contains", func() bool {
		return strings.Contains(data, randomItem())
	})
	runBenchmark("sz.Contains", func() bool {
		return sz.Contains(data, randomItem())
	})
	runBenchmark("strings.Index", func() int {
		return strings.Index(data, randomItem())
	})
	runBenchmark("sz.Index", func() int64 {
		return sz.Index(data, randomItem())
	})
	runBenchmark("strings.LastIndex", func() int {
		return strings.LastIndex(data, randomItem())
	})
	runBenchmark("sz.LastIndex", func() int64 {
		return sz.LastIndex(data, randomItem())
	})
	runBenchmark("strings.IndexAny", func() int {
		return strings.IndexAny(randomItem(), "*^")
	})
	runBenchmark("sz.IndexAny", func() int64 {
		return sz.IndexAny(randomItem(), "*^")
	})
	runBenchmark("strings.Count", func() int {
		return strings.Count(data, randomItem())
	})
	runBenchmark("sz.Count (non-overlap)", func() int64 {
		return sz.Count(data, randomItem(), false)
	})
	runBenchmark("sz.Count (overlap)", func() int64 {
		return sz.Count(data, randomItem(), true)
	})
}
