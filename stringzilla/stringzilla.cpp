#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h> // `mmap`
#include <unistd.h>
#include <fcntl.h>    // `O_RDNNLY`

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stringzilla.hpp"

namespace py = pybind11;
using namespace av::stringzilla;

#if defined(__AVX2__)
using backend_t = speculative_avx2_t;
#elif defined(__ARM_NEON)
using backend_t = speculative_neon_t;
#else
using backend_t = naive_t;
#endif

static constexpr ssize_t ssize_max_k = std::numeric_limits<ssize_t>::max();
span_t to_span(std::string_view s) { return {reinterpret_cast<byte_t const *>(s.data()), s.size()}; }
std::string_view to_stl(span_t s) { return {reinterpret_cast<char const *>(s.begin()), s.size()}; }

span_t slice(span_t span, ssize_t signed_start, ssize_t signed_end = ssize_max_k) {
    if (signed_start < 0 || signed_end < 0)
        throw std::invalid_argument("Negative slices aren't supported yet!");

    signed_end = std::min<ssize_t>(span.len_, signed_end);
    return {span.data_ + signed_start, static_cast<size_t>(signed_end - signed_start)};
    // size_t start = 0;
    // size_t end = span.size();
    // if (signed_start > 0)
    //     start = std::min(static_cast<size_t>(signed_start), end);
    // else if (signed_start < 0)
    //     start = static_cast<ssize_t>(end) < std::abs(signed_start) ? end - static_cast<size_t>(-signed_start) : end;

    // if (signed_end > static_cast<ssize_t>(end))
    //     end = end;
    // else if (signed_end > 0)
    //     end = std::min(end, static_cast<size_t>(signed_end));
    // else if (end < 0)
    //     span = span.after_n(size() - start);

    // size_t start_offset = static_cast<size_t>(start >= 0 ? start : (span.len_ - start));
    // size_t end_offset =
    //     static_cast<size_t>(end > static_cast<ssize_t>(span.len_) ? span.len_ : (end >= 0 ? end : (span.len_ -
    //     end)));
    // return {span.data_ + start_offset, span.len_ - (end_offset - start_offset)};
}

class str_view_t;
class str_t;
class file_t;
class slices_t;

class str_view_t : public std::enable_shared_from_this<str_view_t> {
  protected:
    span_t view_;

  public:
    virtual ~str_view_t() {}
    ssize_t size() const { return static_cast<ssize_t>(view_.size()); }
    bool contains(std::string_view needle, ssize_t start, ssize_t end) const;
    ssize_t find(std::string_view, ssize_t start, ssize_t end) const;
    ssize_t count(std::string_view, ssize_t start, ssize_t end, bool allowoverlap) const;
    std::shared_ptr<slices_t> splitlines(bool keeplinebreaks, char separator) const;
    std::shared_ptr<slices_t> split(std::string_view separator, ssize_t maxsplit, bool keepseparator) const;
    std::shared_ptr<str_view_t> strip(std::string_view characters) const;
};

class str_t : public str_view_t {
    std::string copy_;

  public:
    str_t(std::string string = "") : copy_(string) { view_ = to_span(copy_); }
    ~str_t() {}

    using str_view_t::contains;
    using str_view_t::count;
    using str_view_t::find;
    using str_view_t::size;
    using str_view_t::split;
    using str_view_t::splitlines;
    using str_view_t::strip;
};

class file_t : public str_view_t {
  public:
    file_t(std::string path) {
        struct stat sb;
        int fd = open(path.c_str(), O_RDONLY);
        if (fstat(fd, &sb) != 0)
            throw std::runtime_error("Can't retrieve file size!");
        size_t file_size = sb.st_size;
        void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED)
            throw std::runtime_error("Couldn't map the file!");
        view_ = {reinterpret_cast<byte_t const *>(map), file_size};
    }

    ~file_t() { munmap((void *)view_.data_, view_.len_); }

    using str_view_t::contains;
    using str_view_t::count;
    using str_view_t::find;
    using str_view_t::size;
    using str_view_t::split;
    using str_view_t::splitlines;
    using str_view_t::strip;
};

class slices_t : public std::enable_shared_from_this<slices_t> {
    std::weak_ptr<str_view_t const> whole_;
    std::vector<span_t> parts_;

  public:
    slices_t() = default;
    slices_t(slices_t &&) = default;
    slices_t &operator=(slices_t &&) = default;
    slices_t(std::shared_ptr<str_view_t const> whole, std::vector<span_t> parts)
        : whole_(std::move(whole)), parts_(std::move(parts)) {}

    ssize_t size() const { return static_cast<ssize_t>(parts_.size()); }
    std::string_view operator[](ssize_t i) const {
        return to_stl(i < 0 ? parts_[static_cast<size_t>(size() + i)] : parts_[static_cast<size_t>(i)]);
    }
};

bool str_view_t::contains(std::string_view needle, ssize_t start, ssize_t end) const {

    if (needle.size() == 0)
        return true;
    span_t part = slice(view_, start, end);
    size_t offset = needle.size() == 1 //
                        ? backend_t {}.next_offset(part, static_cast<byte_t>(needle.front()))
                        : backend_t {}.next_offset(part, to_span(needle));
    return offset != part.size();
}

ssize_t str_view_t::find(std::string_view needle, ssize_t start, ssize_t end) const {
    if (needle.size() == 0)
        return 0;
    span_t part = slice(view_, start, end);
    size_t offset = needle.size() == 1 //
                        ? backend_t {}.next_offset(part, static_cast<byte_t>(needle.front()))
                        : backend_t {}.next_offset(part, to_span(needle));
    return offset != part.size() ? offset : -1;
}

ssize_t str_view_t::count(std::string_view needle, ssize_t start, ssize_t end, bool allowoverlap) const {
    if (needle.size() == 0)
        return 0;
    span_t part = slice(view_, start, end);
    auto result = needle.size() == 1 //
                      ? backend_t {}.count(part, static_cast<byte_t>(needle.front()))
                      : backend_t {}.count(part, to_span(needle), allowoverlap);

    return result;
}

std::shared_ptr<slices_t> str_view_t::splitlines(bool keeplinebreaks, char separator) const {

    size_t count_separators = backend_t {}.count(view_, separator);
    std::vector<span_t> parts(count_separators + 1);
    size_t last_start = 0;
    for (size_t i = 0; i != count_separators; ++i) {
        span_t remaining = view_.after_n(last_start);
        size_t offset_in_remaining = backend_t {}.next_offset(remaining, separator);
        parts[i] = span_t {view_.data_ + last_start, offset_in_remaining + keeplinebreaks};
        last_start += offset_in_remaining + 1;
    }
    parts[count_separators] = view_.after_n(last_start);
    return std::make_shared<slices_t>(shared_from_this(), std::move(parts));
}

std::shared_ptr<slices_t> str_view_t::split(std::string_view separator, ssize_t maxsplit, bool keepseparator) const {

    if (separator.size() == 1)
        return splitlines(keepseparator, separator.front());

    std::vector<span_t> parts;
    size_t last_start = 0;
    bool will_continue = true;
    while (last_start < view_.size()) {
        span_t remaining = view_.after_n(last_start);
        size_t offset_in_remaining = backend_t {}.next_offset(remaining, to_span(separator));
        will_continue = offset_in_remaining != remaining.size();
        size_t part_len = offset_in_remaining + separator.size() * keepseparator * will_continue;
        parts.emplace_back(span_t {remaining.begin(), part_len});
        last_start += offset_in_remaining + separator.size();
    }
    // Python marks includes empy ending as well
    if (will_continue)
        parts.emplace_back(span_t {});
    return std::make_shared<slices_t>(shared_from_this(), std::move(parts));
}

std::shared_ptr<str_view_t> str_view_t::strip(std::string_view characters) const { return {}; }

template <typename at>
void define_str_view_ops(py::class_<at, std::shared_ptr<at>> &str_view_class) {

    str_view_class.def( //
        "contains",
        &at::contains,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k);
    str_view_class.def( //
        "find",
        &at::find,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k);
    str_view_class.def( //
        "count",
        &at::count,
        py::arg("needle"),
        py::arg("start") = 0,
        py::arg("end") = ssize_max_k,
        py::arg("allowoverlap") = false);
    str_view_class.def( //
        "splitlines",
        &at::splitlines,
        py::arg("keeplinebreaks") = false,
        py::arg("separator") = '\n');
    str_view_class.def( //
        "split",
        &at::split,
        py::arg("separator") = " ",
        py::arg("maxsplit") = ssize_max_k,
        py::arg("keepseparator") = false);

    str_view_class.def("__len__", &at::size);
    str_view_class.def("__contains__",
                       [](at const &str, std::string_view needle) { return str.contains(needle, 0, ssize_max_k); });
}

PYBIND11_MODULE(compiled, m) {
    m.doc() = "Crunch 100+ GB Strings in Python with ease";

    auto str_class = py::class_<str_t, std::shared_ptr<str_t>>(m, "Str");
    str_class.def(py::init([](std::string arg) { return std::make_shared<str_t>(std::move(arg)); }), py::arg("str"));
    define_str_view_ops(str_class);

    auto file_class = py::class_<file_t, std::shared_ptr<file_t>>(m, "File");
    file_class.def( //
        py::init([](std::string path) { return std::make_shared<file_t>(std::move(path)); }),
        py::arg("path"));
    define_str_view_ops(file_class);

    auto slices_class = py::class_<slices_t, std::shared_ptr<slices_t>>(m, "Slices");
    slices_class.def(py::init([]() { return std::make_shared<slices_t>(); }));
    slices_class.def("__len__", &slices_t::size);
    slices_class.def("__getitem__", &slices_t::operator[], py::arg("index"));
}