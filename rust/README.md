# StringZilla 🦖

StringZilla is the Godzilla of string libraries, splitting, sorting, and shuffling large textual datasets faster than you can say "Tokyo Tower" 😅

- ✅ Single-header pure C 99 implementation [docs](#quick-start-c-🛠️)
- ✅ [Direct CPython bindings](https://ashvardanian.com/posts/pybind11-cpython-tutorial/) with minimal call latency [docs](#quick-start-python-🐍)
- ✅ [SWAR](https://en.wikipedia.org/wiki/SWAR) and [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) acceleration on x86 (AVX2) and ARM (NEON)
- ✅ [Radix](https://en.wikipedia.org/wiki/Radix_sort)-like sorting faster than C++ `std::sort`
- ✅ [Memory-mapping](https://en.wikipedia.org/wiki/Memory-mapped_file) to work with larger-than-RAM datasets
- ✅ Memory-efficient compressed arrays to work with sequences
- 🔜 JavaScript bindings are on their way.

This library saved me tens of thousands of dollars pre-processing large datasets for machine learning, even on the scale of a single experiment.
So if you want to process the 6 Billion images from [LAION](https://laion.ai/blog/laion-5b/), or the 250 Billion web pages from the [CommonCrawl](https://commoncrawl.org/), or even just a few million lines of server logs, and haunted by Python's `open(...).readlines()` and `str().splitlines()` taking forever, this should help 😊

## Performance

StringZilla is built on a very simple heuristic:

> If the first 4 bytes of the string are the same, the strings are likely to be equal.
> Similarly, the first 4 bytes of the strings can be used to determine their relative order most of the time.

Thanks to that it can avoid scalar code processing one `char` at a time and use hyper-scalar code to achieve `memcpy` speeds.
__The implementation fits into a single C 99 header file__ and uses different SIMD flavors and SWAR on older platforms.

### Substring Search

| Backend \ Device         |                    IoT |                   Laptop |                    Server |
| :----------------------- | ---------------------: | -----------------------: | ------------------------: |
| __Speed Comparison__ 🐇   |                        |                          |                           |
| Python `for` loop        |                 4 MB/s |                  14 MB/s |                   11 MB/s |
| C++ `for` loop           |               520 MB/s |                 1.0 GB/s |                  900 MB/s |
| C++ `string.find`        |               560 MB/s |                 1.2 GB/s |                  1.3 GB/s |
| Scalar StringZilla       |                 2 GB/s |                 3.3 GB/s |                  3.5 GB/s |
| Hyper-Scalar StringZilla |           __4.3 GB/s__ |              __12 GB/s__ |             __12.1 GB/s__ |
| __Efficiency Metrics__ 📊 |                        |                          |                           |
| CPU Specs                | 8-core ARM, 0.5 W/core | 8-core Intel, 5.6 W/core | 22-core Intel, 6.3 W/core |
| Performance/Core         |         2.1 - 3.3 GB/s |              __11 GB/s__ |                 10.5 GB/s |
| Bytes/Joule              |           __4.2 GB/J__ |                   2 GB/J |                  1.6 GB/J |

### Split, Partition, Sort, and Shuffle

Coming soon.

## Contributing 👾

Future development plans include:

- [x] [Replace PyBind11 with CPython](https://github.com/ashvardanian/StringZilla/issues/35), [blog](https://ashvardanian.com/posts/pybind11-cpython-tutorial/)
- [x] [Bindings for JavaScript](https://github.com/ashvardanian/StringZilla/issues/25)
- [ ] [Faster string sorting algorithm](https://github.com/ashvardanian/StringZilla/issues/45)
- [ ] [Reverse-order operations in Python](https://github.com/ashvardanian/StringZilla/issues/12)
- [ ] [Splitting with multiple separators at once](https://github.com/ashvardanian/StringZilla/issues/29)
- [ ] Splitting CSV rows into columns
- [ ] UTF-8 validation.
- [ ] Arm SVE backend
- [ ] Bindings for Java and Rust

Here's how to set up your dev environment and run some tests.

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

---

If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], [UStore][ustore], [SimSIMD][simsimd], and [TenPack][tenpack] 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[ustore]: https://github.com/unum-cloud/ustore
[simsimd]: https://github.com/ashvardanian/simsimd
[tenpack]: https://github.com/ashvardanian/tenpack
