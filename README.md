# Stringzilla 🦖

## Crunch 100+ GB Strings in Python with ease, leveraging SIMD Assembly

Stringzilla was born many years ago as a [tutorial for SIMD accelerated string-processing][tutorial].
But one day, processing 100+ GB of Chemistry and AI datasets, I transformed it into a library.
It's designed to replace `open(...).readlines()`, `str().splitlines()`, and many other typical workloads with very long strings.

<table>
<tr>
<td>

<table>
<thead>
<tr>
<th style="text-align:left">Benchmark</th>
<th style="text-align:center">IoT</th>
<th style="text-align:center">Arm Laptop</th>
<th style="text-align:center">x86 Server</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left">Python: <code>str.find</code></td>
<td style="text-align:center">4 MB/s</td>
<td style="text-align:center">14 MB/s</td>
<td style="text-align:center">11 MB/s</td>
</tr>
<tr>
<td style="text-align:left">C++: <code>std::string::find</code></td>
<td style="text-align:center">560 MB/s</td>
<td style="text-align:center">1,2 GB/s</td>
<td style="text-align:center">1,3 GB/s</td>
</tr>
<tr>
<td style="text-align:left">Stringzilla</td>
<td style="text-align:center">4,3 Gb/s</td>
<td style="text-align:center">12 GB/s</td>
<td style="text-align:center">12,1 GB/s</td>
</tr>
</tbody>
</table>

</td>
<td>
<img src="https://github.com/ashvardanian/Stringzilla/blob/main/stringzilla.jpeg?raw=true" height=150px>
</td>
<tr>
</table>

[tutorial]: https://youtu.be/6Sh9QWdzo58

## Usage

```sh
pip install stringzilla
```

There are two classes you can use interchangeably:

```python
from stringzilla import Str, File, Strs

text: str = 'some-string'
text: Str = Str('some-string')
text: File = File('some-file.txt')
```

Once constructed, the following interfaces are supported:

```python
len(text) -> int
'substring' in text -> bool
text[42] -> str
text[42:46] -> str

text.contains(
    'subtring',
    start=0, # optional
    end=9223372036854775807, # optional
) -> bool

text.find(
    'subtring',
    start=0, # optional
    end=9223372036854775807, # optional
) -> int

text.count(
    'subtring',
    start=0, # optional
    end=9223372036854775807, # optional
    **, # non-traditional arguments:
    allowoverlap=False, # optional
) -> int

text.splitlines(
    keeplinebreaks=False, # optional
    **, # non-traditional arguments:
    separator='\n', # optional
) -> Strs # similar to list[str]

text.split(
    separator=' ', # optional
    maxsplit=9223372036854775807, # optional
    **, # non-traditional arguments:
    keepseparator=False, # optional
) -> Strs # similar to list[str]
```

Once split, you can sort, shuffle, and perform other collection-level operations on strings:

```py
lines: Strs = text.split(separator='\n')
lines.sort()
lines.shuffle(seed=42)

sorted_copy: Strs = lines.sorted()
shuffled_copy: Strs = lines.shuffled(seed=42)

lines.append(shuffled_copy.pop(0))
lines.append('Pythonic string')
lines.extend(shuffled_copy)
```

## Development

```sh
rm -rf build && pip install -e . && pytest scripts/test.py -s -x

pip install -e . --no-index --no-deps
```

To benchmark on some custom file and pattern combinations:

```sh
python scripts/bench.py --haystack_path "your file" --needle "your pattern"
```

To benchmark on synthetic data:

```sh
python scripts/bench.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce"
```

To validate packaging:

```sh
cibuildwheel --platform linux
```

Compiling C++ tests:


```sh
brew install libomp llvm
cmake -B ./build_release \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DSTRINGZILLA_USE_OPENMP=1 \
    -DSTRINGZILLA_BUILD_TEST=1 \
    && \
    make -C ./build_release -j && ./build_release/stringzilla_test
```
