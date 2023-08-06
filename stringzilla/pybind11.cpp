
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h> // `stat`
#include <sys/mman.h> // `mmap`
#include <fcntl.h>    // `O_RDNLY`
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <unistd.h> // `ssize_t`
#endif

#include <random>      // `std::random_device`
#include <utility>     // `std::exchange`
#include <limits>      // `std::numeric_limits`
#include <numeric>     // `std::iota`
#include <cmath>       // `std::abs`
#include <algorithm>   // `std::shuffle`
#include <string_view> // `std::string_view`

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stringzilla.h"

namespace py = pybind11;

struct py_span_t;
struct py_str_t;
struct py_file_t;
struct py_subspan_t;
struct py_spans_t;

struct span_t {
    char const *ptr;
    size_t len;

    char const *data() const noexcept { return ptr; }
    size_t size() const noexcept { return len; }
};

static constexpr ssize_t ssize_max_k = std::numeric_limits<ssize_t>::max();
static constexpr size_t size_max_k = std::numeric_limits<size_t>::max();

inline size_t find_substr(span_t h_span, char n) noexcept {
    strzl_haystack_t h {h_span.ptr, h_span.len};
    return strzl_naive_find_char(h, n);
}

inline size_t find_substr(span_t h_span, span_t n_span) noexcept {
    strzl_haystack_t h {h_span.ptr, h_span.len};
    strzl_needle_t n {n_span.ptr, n_span.len, 0};

#if defined(__AVX2__)
    return strzl_avx2_find_substr(h, n);
#elif defined(__ARM_NEON)
    return strzl_neon_find_substr(h, n);
#else
    return strzl_naive_find_substr(h, n);
#endif
}

inline size_t count_char(span_t h_span, char n) noexcept {
    strzl_haystack_t h {h_span.ptr, h_span.len};
    return strzl_naive_count_char(h, n);
}

inline size_t count_substr(span_t h, span_t n, bool overlap = false) noexcept {

    if (n.len == 1)
        return count_char(h, *n.ptr);
    if (h.len < n.len)
        return 0;

    size_t result = 0;
    if (overlap) {
        while (h.len) {
            size_t offset = find_substr(h, n);
            result += offset != h.len;
            h.ptr += offset;
            h.len -= offset;
        }
    }

    else {
        while (h.len) {
            size_t offset = find_substr(h, n);
            result += offset != h.len;
            h.ptr += offset + n.len;
            h.len -= offset;
            h.len -= n.len * bool(h.len);
        }
    }

    return result;
}

span_t to_span(std::string_view s) { return {s.data(), s.size()}; }
std::string_view to_stl(span_t s) { return {s.data(), s.size()}; }

struct index_span_t {
    size_t offset;
    size_t length;
};

index_span_t slice(size_t length, ssize_t start, ssize_t end) {
    ssize_t len = static_cast<ssize_t>(length);
    ssize_t abs_start = std::abs(start);
    ssize_t abs_end = std::abs(end);

    if (len == 0 || start == end)
        return {0ul, 0ul};

    if (start > end) {
        if ((start < 0 && end < 0) || (start >= 0 && end > 0) || len - abs_end < start)
            return {0ul, 0ul};
        end = len - abs_end;
    }
    else if (start < 0 && end < 0) {
        if (abs_start <= len && abs_end <= len) {
            start = len + start;
            end = len + end;
        }
        else if (abs_start > len && abs_end <= len) {
            start = 0;
            end = len + end;
        }
        else if (abs_start <= len && abs_end > len) {
            start = len + start;
            end = len;
        }
        else if (abs_start > len && abs_end > len) {
            start = 0;
            end = len;
        }
    }
    else if (start < 0 && end >= 0) {
        end = end == 0 ? len : std::min(end, len);
        if (!((start = len - abs_start) < end && start >= 0))
            start = end = 0;
    }
    else if (start >= 0 && end < 0) {
        if (len >= start) {
            if ((len + end) >= start)
                end = len + end;
            else
                end = len;
        }
        else
            end = start;
    }
    else {
        start = std::min(start, len);
        end = end == 0 ? len : std::min(end, len);
    }
    return {static_cast<size_t>(start), static_cast<size_t>(end - start)};
}

size_t unsigned_offset(size_t length, ssize_t idx) {
    if (idx >= 0) {
        if (static_cast<size_t>(idx) > length)
            throw std::out_of_range("Accessing beyond content length");
        return static_cast<size_t>(idx);
    }
    else {
        if (static_cast<size_t>(-idx) > length)
            throw std::out_of_range("Accessing beyond content length");
        return static_cast<size_t>(length + idx);
    }
}

span_t subspan(span_t span, ssize_t start, ssize_t end = ssize_max_k) {
    index_span_t index_span = slice(span.size(), start, end);
    return {span.ptr + index_span.offset, index_span.length};
}

struct py_span_t : public span_t, public std::enable_shared_from_this<py_span_t> {

    py_span_t(span_t view = {}) : span_t(view) {}
    virtual ~py_span_t() {}

    using span_t::len;
    using span_t::ptr;

    span_t span() const { return {ptr, len}; }
    ssize_t size() const { return static_cast<ssize_t>(len); }
    bool contains(std::string_view needle, ssize_t start, ssize_t end) const;
    ssize_t find(std::string_view, ssize_t start, ssize_t end) const;
    ssize_t count(std::string_view, ssize_t start, ssize_t end, bool allowoverlap) const;
    std::shared_ptr<py_spans_t> splitlines(bool keeplinebreaks, char separator, size_t maxsplit) const;
    std::shared_ptr<py_spans_t> split(std::string_view separator, size_t maxsplit, bool keepseparator) const;
    std::shared_ptr<py_subspan_t> sub(ssize_t start, ssize_t end) const;

    char const *begin() const { return reinterpret_cast<char const *>(ptr); }
    char const *end() const { return begin() + len; }
    char at(ssize_t offset) const { return begin()[unsigned_offset(len, offset)]; }
    py::str to_python() const { return {begin(), len}; }

    bool operator==(py::str const &str) const { return to_stl({ptr, len}) == str.cast<std::string_view>(); }
    bool operator!=(py::str const &str) const { return !(*this == str); }
    bool operator==(py_span_t const &other) const { return to_stl({ptr, len}) == to_stl({other.ptr, other.len}); }
    bool operator!=(py_span_t const &other) const { return !(*this == other); }
    bool operator>(py::str const &str) const { return to_stl({ptr, len}) > str.cast<std::string_view>(); }
    bool operator<(py::str const &str) const { return to_stl({ptr, len}) < str.cast<std::string_view>(); }
    bool operator>(py_span_t const &other) const { return to_stl({ptr, len}) > to_stl({other.ptr, other.len}); }
    bool operator<(py_span_t const &other) const { return to_stl({ptr, len}) < to_stl({other.ptr, other.len}); }

    span_t after_n(size_t offset) const noexcept {
        return (offset < len) ? span_t {ptr + offset, len - offset} : span_t {};
    }
    span_t before_n(size_t tail) const noexcept {
        return (tail < len) ? span_t {ptr + len - tail, len - tail} : span_t {};
    }
};

struct py_str_t : public py_span_t {
    std::string copy_;

    py_str_t(std::string string = "") : copy_(string) { ptr = to_span(copy_).ptr, len = to_span(copy_).len; }
    ~py_str_t() {}

    using py_span_t::contains;
    using py_span_t::count;
    using py_span_t::find;
    using py_span_t::size;
    using py_span_t::split;
    using py_span_t::splitlines;
};

struct py_file_t : public py_span_t {
    std::string path;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    HANDLE file_handle = nullptr;
    HANDLE mapping_handle = nullptr;
#else
    int file_descriptor = 0;
#endif

  public:
    py_file_t(std::string const &path) { open(path); }
    ~py_file_t() { close(); }

    void reopen() { open(path); }
    void open(std::string const &p) {
        close();
        path = p;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)

        file_handle =
            CreateFile(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        if (file_handle == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Couldn't map the file!");

        mapping_handle = CreateFileMapping(file_handle, 0, PAGE_READONLY, 0, 0, 0);
        if (mapping_handle == 0) {
            CloseHandle(std::exchange(file_handle, nullptr));
            throw std::runtime_error("Couldn't map the file!");
        }

        char *file = (char *)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (file == 0) {
            CloseHandle(std::exchange(mapping_handle, nullptr));
            CloseHandle(std::exchange(file_handle, nullptr));
            throw std::runtime_error("Couldn't map the file!");
        }
        ptr = file;
        len = GetFileSize(file_handle, 0);
#else
        struct stat sb;
        file_descriptor = ::open(path.c_str(), O_RDONLY);
        if (fstat(file_descriptor, &sb) != 0) {
            ::close(std::exchange(file_descriptor, 0));
            throw std::runtime_error("Can't retrieve file size!");
        }
        size_t file_size = sb.st_size;
        void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, file_descriptor, 0);
        if (map == MAP_FAILED) {
            ::close(std::exchange(file_descriptor, 0));
            throw std::runtime_error("Couldn't map the file!");
        }
        ptr = reinterpret_cast<char const *>(map);
        len = file_size;
#endif
    }

    void close() {

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        if (ptr)
            UnmapViewOfFile(std::exchange(ptr, nullptr)), len = 0;
        if (mapping_handle)
            CloseHandle(std::exchange(mapping_handle, nullptr));
        if (file_handle)
            CloseHandle(std::exchange(file_handle, nullptr));

#else
        if (ptr)
            munmap((void *)std::exchange(ptr, nullptr), std::exchange(len, 0));
        if (file_descriptor != 0)
            ::close(std::exchange(file_descriptor, 0));
#endif
    }

    using py_span_t::contains;
    using py_span_t::count;
    using py_span_t::find;
    using py_span_t::size;
    using py_span_t::split;
    using py_span_t::splitlines;
};

struct py_subspan_t : public py_span_t {
    std::shared_ptr<py_span_t const> parent_;

  public:
    py_subspan_t() = default;
    py_subspan_t(py_subspan_t &&) = default;
    py_subspan_t &operator=(py_subspan_t &&) = default;
    py_subspan_t(std::shared_ptr<py_span_t const> parent, span_t str) : parent_(std::move(parent)) {
        ptr = str.ptr, len = str.len;
    }

    using py_span_t::contains;
    using py_span_t::count;
    using py_span_t::find;
    using py_span_t::size;
    using py_span_t::split;
    using py_span_t::splitlines;
};

struct py_spans_t : public std::enable_shared_from_this<py_spans_t> {
    std::shared_ptr<py_span_t const> whole_;
    std::vector<span_t> parts_;

    static char const *strzl_array_get_begin(void const *raw, size_t i) { return ((span_t *)raw)[i].ptr; }
    static size_t strzl_array_get_length(void const *raw, size_t i) { return ((span_t *)raw)[i].len; }

  public:
    py_spans_t() = default;
    py_spans_t(py_spans_t &&) = default;
    py_spans_t &operator=(py_spans_t &&) = default;
    py_spans_t(std::shared_ptr<py_span_t const> whole, std::vector<span_t> parts)
        : whole_(std::move(whole)), parts_(std::move(parts)) {}

    struct iterator_t {
        py_spans_t const *py_spans_ = nullptr;
        size_t idx_ = 0;

        bool operator==(iterator_t const &other) const { return idx_ == other.idx_; }
        bool operator!=(iterator_t const &other) const { return idx_ != other.idx_; }
        std::shared_ptr<py_subspan_t> operator*() const { return py_spans_->at(idx_); }
        iterator_t &operator++() {
            idx_++;
            return *this;
        }
        iterator_t operator++(int) {
            iterator_t old(*this);
            ++*this;
            return old;
        }
    };

    std::shared_ptr<py_subspan_t> at(ssize_t i) const {
        return std::make_shared<py_subspan_t>(whole_, parts_[unsigned_offset(size(), i)]);
    }

    std::shared_ptr<py_spans_t> sub(ssize_t start, ssize_t end, ssize_t step, ssize_t length) const {
        if (step == 1) {
            auto first_part_it = parts_.begin() + start;
            std::vector<span_t> sub_parts(first_part_it, first_part_it + length);
            return std::make_shared<py_spans_t>(whole_, std::move(sub_parts));
        }
        std::vector<span_t> sub_parts(length);
        for (ssize_t parts_idx = start, sub_idx = 0; sub_idx < length; parts_idx += step, ++sub_idx)
            sub_parts[sub_idx] = parts_[parts_idx];
        return std::make_shared<py_spans_t>(whole_, std::move(sub_parts));
    }

    iterator_t begin() const { return {this, 0}; }
    iterator_t end() const { return {this, parts_.size()}; }
    ssize_t size() const { return static_cast<ssize_t>(parts_.size()); }

    void sort() {
        std::vector<std::size_t> permute(parts_.size());
        std::iota(permute.begin(), permute.end(), 0ul);
        strzl_array_t array;
        array.order = permute.data();
        array.count = permute.size();
        array.handle = parts_.data();
        array.get_begin = strzl_array_get_begin;
        array.get_length = strzl_array_get_length;
        strzl_sort(&array, nullptr);
        std::vector<span_t> new_parts(parts_.size());
        for (std::size_t i = 0; i != parts_.size(); ++i)
            new_parts[i] = parts_[permute[i]];
        parts_ = new_parts;
    }

    void shuffle(std::optional<std::size_t> maybe_seed) {
        std::random_device device;
        std::size_t seed = maybe_seed ? *maybe_seed : device();
        std::mt19937 generator {seed};
        std::shuffle(parts_.begin(), parts_.end(), generator);
    }
};

bool py_span_t::contains(std::string_view needle, ssize_t start, ssize_t end) const {
    if (needle.size() == 0)
        return true;
    span_t part = subspan(span(), start, end);
    size_t offset = needle.size() == 1 //
                        ? find_substr(part, needle.front())
                        : find_substr(part, to_span(needle));
    return offset != part.len;
}

ssize_t py_span_t::find(std::string_view needle, ssize_t start, ssize_t end) const {
    if (needle.size() == 0)
        return 0;
    span_t part = subspan(span(), start, end);
    size_t offset = needle.size() == 1 //
                        ? find_substr(part, needle.front())
                        : find_substr(part, to_span(needle));
    return offset != part.len ? offset : -1;
}

ssize_t py_span_t::count(std::string_view needle, ssize_t start, ssize_t end, bool allowoverlap) const {
    if (needle.size() == 0)
        return 0;
    span_t part = subspan(span(), start, end);
    auto result = needle.size() == 1 //
                      ? count_char(part, needle.front())
                      : count_substr(part, to_span(needle), allowoverlap);
    return result;
}

std::shared_ptr<py_spans_t> py_span_t::splitlines(bool keeplinebreaks, char separator, size_t maxsplit) const {

    size_t count_separators = count_char(span(), separator);
    std::vector<span_t> parts(std::min(count_separators + 1, maxsplit));
    size_t last_start = 0;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        span_t remaining = after_n(last_start);
        size_t offset_in_remaining = find_substr(remaining, separator);
        parts[i] = span_t {ptr + last_start, offset_in_remaining + keeplinebreaks};
        last_start += offset_in_remaining + 1;
    }
    parts[count_separators] = after_n(last_start);
    return std::make_shared<py_spans_t>(shared_from_this(), std::move(parts));
}

std::shared_ptr<py_spans_t> py_span_t::split(std::string_view separator, size_t maxsplit, bool keepseparator) const {

    if (separator.size() == 1 && maxsplit == ssize_max_k)
        return splitlines(keepseparator, separator.front(), maxsplit);

    std::vector<span_t> parts;
    size_t last_start = 0;
    bool will_continue = true;
    while (last_start < len && parts.size() + 1 < maxsplit) {
        span_t remaining = after_n(last_start);
        size_t offset_in_remaining = find_substr(remaining, to_span(separator));
        will_continue = offset_in_remaining != remaining.size();
        size_t part_len = offset_in_remaining + separator.size() * keepseparator * will_continue;
        parts.emplace_back(span_t {remaining.data(), part_len});
        last_start += offset_in_remaining + separator.size();
    }
    // Python marks includes empy ending as well
    if (will_continue)
        parts.emplace_back(after_n(last_start));
    return std::make_shared<py_spans_t>(shared_from_this(), std::move(parts));
}

std::shared_ptr<py_subspan_t> py_span_t::sub(ssize_t start, ssize_t end) const {
    index_span_t index_span = slice(size(), start, end);
    return std::make_shared<py_subspan_t>(shared_from_this(), span_t {ptr + index_span.offset, index_span.length});
}

template <typename at>
void define_comparsion_ops(py::class_<at, std::shared_ptr<at>> &str_view_struct) {
    str_view_struct.def("__eq__", [](at const &self, py::str const &str) { return self == str; });
    str_view_struct.def("__ne__", [](at const &self, py::str const &str) { return self != str; });
    str_view_struct.def("__eq__", [](at const &self, at const &other) { return self == other; });
    str_view_struct.def("__ne__", [](at const &self, at const &other) { return self != other; });
    str_view_struct.def("__gt__", [](at const &self, py::str const &str) { return self > str; });
    str_view_struct.def("__lt__", [](at const &self, py::str const &str) { return self < str; });
    str_view_struct.def("__gt__", [](at const &self, at const &other) { return self > other; });
    str_view_struct.def("__lt__", [](at const &self, at const &other) { return self < other; });
}

template <typename at>
void define_slice_ops(py::class_<at, std::shared_ptr<at>> &str_view_struct) {

    str_view_struct.def( //
        "contains",
        &at::contains,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k,
        py::call_guard<py::gil_scoped_release>());
    str_view_struct.def( //
        "find",
        &at::find,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k,
        py::call_guard<py::gil_scoped_release>());
    str_view_struct.def( //
        "count",
        &at::count,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k,
        py::arg("allowoverlap") = false,
        py::call_guard<py::gil_scoped_release>());
    str_view_struct.def( //
        "splitlines",
        &at::splitlines,
        py::arg("keeplinebreaks") = false,
        py::arg("separator") = '\n',
        py::kw_only(),
        py::arg("maxsplit") = size_max_k,
        py::call_guard<py::gil_scoped_release>());
    str_view_struct.def( //
        "split",
        &at::split,
        py::arg("separator") = " ",
        py::arg("maxsplit") = size_max_k,
        py::kw_only(),
        py::arg("keepseparator") = false,
        py::call_guard<py::gil_scoped_release>());
    str_view_struct.def( //
        "sub",
        &at::sub,
        py::arg("start") = 0,
        py::arg("end") = 0);

    // Substring presence operator
    str_view_struct.def("__contains__",
                        [](at const &str, std::string_view needle) { return str.contains(needle, 0, ssize_max_k); });

    // Character access operators
    str_view_struct.def("__str__", &at::to_python);
    str_view_struct.def("__getitem__", &at::at, py::arg("index"));
    str_view_struct.def("__len__", &at::size);
    str_view_struct.def(
        "__iter__",
        [](at const &s) { return py::make_iterator(s.begin(), s.end()); },
        py::keep_alive<0, 1>());
}

PYBIND11_MODULE(stringzilla, m) {
    m.doc() = "Crunch 100+ GB Strings in Python with ease";

    auto py_span = py::class_<py_span_t, std::shared_ptr<py_span_t>>(m, "Span");
    define_comparsion_ops(py_span);
    define_slice_ops(py_span);

    auto py_subspan = py::class_<py_subspan_t, std::shared_ptr<py_subspan_t>>(m, "SubSpan");
    define_comparsion_ops(py_subspan);
    define_slice_ops(py_subspan);

    auto py_str = py::class_<py_str_t, std::shared_ptr<py_str_t>>(m, "Str");
    py_str.def(py::init([](std::string arg) { return std::make_shared<py_str_t>(std::move(arg)); }), py::arg("str"));
    py_str.def("__getitem__", [](py_str_t &s, py::slice slice) {
        ssize_t start, stop, step, length;
        if (!slice.compute(s.size(), &start, &stop, &step, &length))
            throw py::error_already_set();
        if (step != 1)
            throw std::invalid_argument("Step argument is not supported for Str");
        return s.sub(start, stop);
    });
    define_comparsion_ops(py_str);
    define_slice_ops(py_str);

    auto py_file = py::class_<py_file_t, std::shared_ptr<py_file_t>>(m, "File");
    py_file.def( //
        py::init([](std::string path) { return std::make_shared<py_file_t>(std::move(path)); }),
        py::arg("path"));
    define_slice_ops(py_file);
    py_file.def("open", &py_file_t::open, py::arg("path"));
    py_file.def("open", &py_file_t::reopen);
    py_file.def("close", &py_file_t::close);
    py_file.def("__getitem__", [](py_file_t &s, py::slice slice) {
        ssize_t start, stop, step, length;
        if (!slice.compute(s.size(), &start, &stop, &step, &length))
            throw py::error_already_set();
        if (step != 1)
            throw std::invalid_argument("Step argument is not supported for File");
        return s.sub(start, stop);
    });

    auto py_strs = py::class_<py_spans_t, std::shared_ptr<py_spans_t>>(m, "Strs");
    py_strs.def(py::init([]() { return std::make_shared<py_spans_t>(); }));
    py_strs.def("__len__", &py_spans_t::size);
    py_strs.def("__getitem__", &py_spans_t::at, py::arg("index"));
    py_strs.def(
        "__iter__",
        [](py_spans_t const &s) { return py::make_iterator(s.begin(), s.end()); },
        py::keep_alive<0, 1>());
    py_strs.def("sort", &py_spans_t::sort, py::call_guard<py::gil_scoped_release>());
    py_strs.def("shuffle",
                &py_spans_t::shuffle,
                py::arg("seed") = std::nullopt,
                py::call_guard<py::gil_scoped_release>());
    py_strs.def("__getitem__", [](py_spans_t &s, py::slice slice) {
        ssize_t start, stop, step, length;
        if (!slice.compute(s.size(), &start, &stop, &step, &length))
            throw py::error_already_set();
        return s.sub(start, stop, step, length);
    });
    py_strs.def( //
        "sub",
        [](py_spans_t &s, ssize_t start, ssize_t stop, ssize_t step = 1) {
            auto index_span = slice(s.size(), start, stop);
            ssize_t length = stop = index_span.length;
            start = index_span.offset;
            return s.sub(start, stop, step, length);
        });
}