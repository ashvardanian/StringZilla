//
//  swift/StringProtocol+StringZilla.swift
//  Extension of StringProtocol to interface with StringZilla functionalities.
//
//  Docs:
//  - Accessing immutable UTF8-range:
//    https://developer.apple.com/documentation/swift/string/utf8view
//
//  More reading materials:
//  - String’s ABI and UTF-8. Nov 2018
//    https://forums.swift.org/t/string-s-abi-and-utf-8/17676
//  - Stable pointer into a C string without copying it? Aug 2021
//    https://forums.swift.org/t/stable-pointer-into-a-c-string-without-copying-it/51244/1
//
//  - Author: Ash Vardanian
//  - Date: January 21, 2024
//

import StringZillaC

/// Result of a three-way string comparison.
///
/// Mirrors `Foundation.ComparisonResult` but is defined here so the package stays
/// Foundation-free and usable on Linux and embedded targets. The cases match the verdicts of the
/// exact and case-insensitive orderings.
public enum StringZillaOrdering: Sendable {
    case ascending  // The receiver sorts before the argument.
    case equal  // The two compare as equal.
    case descending  // The receiver sorts after the argument.
}

/// Unicode normalization form selector.
public enum StringZillaNormalizationForm: RawRepresentable, Sendable {
    case nfd  // Canonical decomposition.
    case nfc  // Canonical decomposition + canonical composition.
    case nfkd  // Compatibility decomposition.
    case nfkc  // Compatibility decomposition + canonical composition.

    public typealias RawValue = sz_normal_form_t

    public var rawValue: sz_normal_form_t {
        switch self {
        case .nfd: return sz_normal_form_nfd_k
        case .nfc: return sz_normal_form_nfc_k
        case .nfkd: return sz_normal_form_nfkd_k
        case .nfkc: return sz_normal_form_nfkc_k
        }
    }

    public init?(rawValue: sz_normal_form_t) {
        switch rawValue {
        case sz_normal_form_nfd_k: self = .nfd
        case sz_normal_form_nfc_k: self = .nfc
        case sz_normal_form_nfkd_k: self = .nfkd
        case sz_normal_form_nfkc_k: self = .nfkc
        default: return nil
        }
    }
}

/// Protocol defining a single-byte data type.
private protocol SingleByte {}

extension UInt8: SingleByte {}
extension Int8: SingleByte {}  // This would match `CChar` as well.

/// Passes UTF-8 code units to `body` as a C pointer and length, lending a valid pointer even for
/// an empty buffer, whose `baseAddress` may be `nil`.
@inlinable
func withStringZillaBuffer<R>(
    _ buffer: UnsafeBufferPointer<UInt8>,
    _ body: (sz_cptr_t, sz_size_t) throws -> R
) rethrows -> R {
    guard let baseAddress = buffer.baseAddress else { return try "".withCString { try body($0, 0) } }
    return try body(UnsafeRawPointer(baseAddress).assumingMemoryBound(to: CChar.self), sz_size_t(buffer.count))
}

/// Traps when a call reports a failure, which no CPU call can, as every enabled set keeps serial.
func stringZillaCheck(_ status: sz_status_t) {
    precondition(status == sz_success_k, String(cString: sz_status_name(status)))
}

/// Collects the UTF-8 code units of `characters` into a byte set, complemented when `inverted`.
private func stringZillaByteset<S: StringZillaViewable>(_ characters: S, inverted: Bool) -> sz_byteset_t {
    var set = sz_byteset_t()
    characters.withStringZillaScope { pointer, length in
        let bytes = UnsafeRawPointer(pointer).assumingMemoryBound(to: UInt8.self)
        for offset in 0 ..< Int(length) { sz_byteset_add_u8(&set, bytes[offset]) }
    }
    if inverted { sz_byteset_invert(&set) }
    return set
}

/// Protocol defining the interface for StringZilla-compatible byte-spans.
///
/// # Discussion:
/// The Swift documentation is extremely vague about the actual memory layout of a String
/// and the cost of obtaining the underlying UTF8 representation or any other raw pointers.
/// https://developer.apple.com/documentation/swift/stringprotocol/withcstring(_:)
/// https://developer.apple.com/documentation/swift/stringprotocol/withcstring(encodedas:_:)
/// https://developer.apple.com/documentation/swift/stringprotocol/data(using:allowlossyconversion:)
public protocol StringZillaViewable: Collection {
    /// A type that represents a position in the collection.
    ///
    /// Executes a closure with a pointer to the string's UTF8 C representation and its length.
    ///
    /// - Parameters:
    ///   - body: A closure that takes a pointer to a C string and its length.
    /// - Throws: Can throw an error.
    /// - Returns: Returns a value of type R, which is the result of the closure.
    func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    ///
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer, as obtained from `withStringZillaScope`.
    /// - Returns: The calculated index offset.
    func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index
}

extension String: StringZillaViewable {
    public typealias Index = String.Index

    @_transparent
    public func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R {
        let cLength = sz_size_t(utf8.count)
        return try withCString { cString in
            try body(cString, cLength)
        }
    }

    @_transparent
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t)
        -> Index
    {
        utf8.index(utf8.startIndex, offsetBy: bytePointer - startPointer)
    }
}

extension Substring.UTF8View: StringZillaViewable {
    public typealias Index = Substring.UTF8View.Index

    /// Executes a closure with a pointer to the UTF8View's contiguous storage of single-byte
    /// elements, the UTF-8 code units, copying them first when the storage is not contiguous,
    /// as for a lazily bridged `NSString`.
    /// - Parameters:
    ///   - body: A closure that takes a pointer to the contiguous storage and its size.
    @_transparent
    public func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R {
        if let result = try withContiguousStorageIfAvailable({ try withStringZillaBuffer($0, body) }) { return result }
        return try Array(self).withUnsafeBufferPointer { try withStringZillaBuffer($0, body) }
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer, as obtained from `withStringZillaScope`.
    /// - Returns: The calculated index offset.
    @_transparent
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t)
        -> Index
    {
        return index(startIndex, offsetBy: bytePointer - startPointer)
    }
}

extension String.UTF8View: StringZillaViewable {
    public typealias Index = String.UTF8View.Index

    /// Executes a closure with a pointer to the UTF8View's contiguous storage of single-byte
    /// elements, the UTF-8 code units, copying them first when the storage is not contiguous,
    /// as for a lazily bridged `NSString`.
    /// - Parameters:
    ///   - body: A closure that takes a pointer to the contiguous storage and its size.
    public func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R {
        if let result = try withContiguousStorageIfAvailable({ try withStringZillaBuffer($0, body) }) { return result }
        return try Array(self).withUnsafeBufferPointer { try withStringZillaBuffer($0, body) }
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer, as obtained from `withStringZillaScope`.
    /// - Returns: The calculated index offset.
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t)
        -> Index
    {
        return index(startIndex, offsetBy: bytePointer - startPointer)
    }
}

extension StringZillaViewable {
    /// Computes a 64-bit hash of the string content using StringZilla's fast hash algorithm.
    /// - Parameter seed: Optional seed value for the hash function, defaulting to 0.
    /// - Returns: A 64-bit unsigned integer hash value.
    public func hash(seed: UInt64 = 0) -> UInt64 {
        var hash = sz_u64_t()
        withStringZillaScope { pointer, length in
            stringZillaCheck(sz_hash_best(pointer, length, sz_u64_t(seed), &hash, Device.cpuEnabled.native, nil))
        }
        return UInt64(hash)
    }

    /// Counts the Unicode codepoints in the receiver's UTF-8 bytes.
    /// - Returns: The number of codepoints, matching `unicodeScalars.count` on well-formed input.
    @_specialize(where Self == String)
    @_specialize(where Self == String.UTF8View)
    public func countRunes() -> Int {
        var count: sz_size_t = 0
        withStringZillaScope { pointer, length in
            stringZillaCheck(sz_utf8_count_best(pointer, length, &count, Device.cpuEnabled.native, nil))
        }
        return Int(count)
    }

    /// Resolves a zero-based codepoint index to a position in the receiver.
    /// - Parameter runeIndex: The zero-based codepoint index to resolve.
    /// - Returns: The index of that codepoint, or `nil` if the receiver holds fewer.
    @_specialize(where Self == String)
    @_specialize(where Self == String.UTF8View)
    public func index(ofRune runeIndex: Int) -> Index? {
        guard runeIndex >= 0 else { return nil }
        var result: Index?
        withStringZillaScope { pointer, length in
            var position: sz_cptr_t?
            stringZillaCheck(
                sz_utf8_seek_best(pointer, length, sz_size_t(runeIndex), &position, Device.cpuEnabled.native, nil)
            )
            if let runePointer = position {
                result = self.stringZillaByteOffset(forByte: runePointer, after: pointer)
            }
        }
        return result
    }

    /// Finds the first occurrence of the specified substring within the receiver.
    /// - Parameter needle: The substring to search for.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findFirst<S: StringZillaViewable>(substring needle: S) -> Index? {
        var result: Index?
        withStringZillaScope { hPointer, hLength in
            needle.withStringZillaScope { nPointer, nLength in
                var match: sz_cptr_t?
                stringZillaCheck(
                    sz_find_best(hPointer, hLength, nPointer, nLength, &match, Device.cpuEnabled.native, nil)
                )
                if let matchPointer = match {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the last occurrence of the specified substring within the receiver.
    /// - Parameter needle: The substring to search for.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findLast<S: StringZillaViewable>(substring needle: S) -> Index? {
        var result: Index?
        withStringZillaScope { hPointer, hLength in
            needle.withStringZillaScope { nPointer, nLength in
                var match: sz_cptr_t?
                stringZillaCheck(
                    sz_rfind_best(hPointer, hLength, nPointer, nLength, &match, Device.cpuEnabled.native, nil)
                )
                if let matchPointer = match {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the first occurrence of the specified character-set members within the receiver.
    /// - Parameter characters: A string-like collection of characters to match.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findFirst<S: StringZillaViewable>(characterFrom characters: S) -> Index? {
        var result: Index?
        var set = stringZillaByteset(characters, inverted: false)
        withStringZillaScope { hPointer, hLength in
            var match: sz_cptr_t?
            stringZillaCheck(sz_find_byteset_best(hPointer, hLength, &set, &match, Device.cpuEnabled.native, nil))
            if let matchPointer = match {
                result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
            }
        }
        return result
    }

    /// Finds the last occurrence of the specified character-set members within the receiver.
    /// - Parameter characters: A string-like collection of characters to match.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findLast<S: StringZillaViewable>(characterFrom characters: S) -> Index? {
        var result: Index?
        var set = stringZillaByteset(characters, inverted: false)
        withStringZillaScope { hPointer, hLength in
            var match: sz_cptr_t?
            stringZillaCheck(sz_rfind_byteset_best(hPointer, hLength, &set, &match, Device.cpuEnabled.native, nil))
            if let matchPointer = match {
                result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
            }
        }
        return result
    }

    /// Finds the first occurrence of a character outside of the the given character-set
    /// within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findFirst<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
        var result: Index?
        var set = stringZillaByteset(characters, inverted: true)
        withStringZillaScope { hPointer, hLength in
            var match: sz_cptr_t?
            stringZillaCheck(sz_find_byteset_best(hPointer, hLength, &set, &match, Device.cpuEnabled.native, nil))
            if let matchPointer = match {
                result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
            }
        }
        return result
    }

    /// Finds the last occurrence of a character outside of the the given character-set
    /// within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findLast<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
        var result: Index?
        var set = stringZillaByteset(characters, inverted: true)
        withStringZillaScope { hPointer, hLength in
            var match: sz_cptr_t?
            stringZillaCheck(sz_rfind_byteset_best(hPointer, hLength, &set, &match, Device.cpuEnabled.native, nil))
            if let matchPointer = match {
                result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
            }
        }
        return result
    }

    /// Applies full Unicode case folding to the content's UTF-8 bytes.
    /// The returned bytes are UTF-8 and may be longer than the input, as "ß" → "ss" is.
    public func utf8UncasedFoldedBytes() -> [UInt8] {
        var folded: [UInt8] = []
        withStringZillaScope { pointer, length in
            if length == 0 {
                folded = []
                return
            }
            let capacity = Int(length) * 3
            var destination = [UInt8](repeating: 0, count: capacity)
            var outLen: sz_size_t = 0
            destination.withUnsafeMutableBufferPointer { bufferPointer in
                stringZillaCheck(
                    sz_utf8_uncased_fold_best(
                        pointer,
                        length,
                        bufferPointer.baseAddress,
                        &outLen,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
            }
            let actual = Int(outLen)
            if actual < destination.count { destination.removeLast(destination.count - actual) }
            folded = destination
        }
        return folded
    }

    /// Produces the UTF-8 bytes of the receiver after applying a Unicode normalization form.
    /// The output may be longer than the input; the worst-case expansion is 18× per source
    /// byte, under NFKD.
    /// - Parameter form: The normalization form to apply, defaulting to `.nfc`.
    /// - Returns: The normalized UTF-8 bytes.
    public func utf8Normalized(_ form: StringZillaNormalizationForm = .nfc) -> [UInt8] {
        var normalized: [UInt8] = []
        withStringZillaScope { pointer, length in
            if length == 0 {
                normalized = []
                return
            }
            let capacity = Int(length) * 18
            var destination = [UInt8](repeating: 0, count: capacity)
            var outLen: sz_size_t = 0
            destination.withUnsafeMutableBufferPointer { bufferPointer in
                stringZillaCheck(
                    sz_utf8_norm_best(
                        pointer,
                        length,
                        form.rawValue,
                        bufferPointer.baseAddress,
                        &outLen,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
            }
            let actual = Int(outLen)
            if actual < destination.count { destination.removeLast(destination.count - actual) }
            normalized = destination
        }
        return normalized
    }

    /// Returns the index of the first byte violating the given normalization form, or `nil`
    /// if already normalized.
    /// - Parameter form: The normalization form to test.
    /// - Returns: The `Index` of the first non-conforming byte, or `nil` if the content is already
    ///   in `form`.
    public func utf8NormalizationViolation(_ form: StringZillaNormalizationForm) -> Index? {
        var result: Index?
        withStringZillaScope { pointer, length in
            var violation: sz_cptr_t?
            stringZillaCheck(
                sz_utf8_find_denormalized_best(
                    pointer,
                    length,
                    form.rawValue,
                    &violation,
                    Device.cpuEnabled.native,
                    nil
                )
            )
            if let violationPointer = violation {
                result = self.stringZillaByteOffset(forByte: violationPointer, after: pointer)
            }
        }
        return result
    }

    /// Returns `true` if the content is already in the given Unicode normalization form.
    /// - Parameter form: The normalization form to test.
    /// - Returns: `true` when no normalization violation is found.
    public func isUtf8Normalized(_ form: StringZillaNormalizationForm) -> Bool {
        utf8NormalizationViolation(form) == nil
    }

    /// Finds the first uncased occurrence of `needle` using full Unicode case folding.
    /// Returns a byte-accurate range into the receiver.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func utf8UncasedFind<S: StringZillaViewable>(substring needle: S) -> Range<Index>? {
        var result: Range<Index>?
        withStringZillaScope { hPointer, hLength in
            needle.withStringZillaScope { nPointer, nLength in
                var prepared = sz_utf8_uncased_needle_t()
                stringZillaCheck(
                    sz_utf8_uncased_needle_init_best(nPointer, nLength, &prepared, Device.cpuEnabled.native, nil)
                )
                var match: sz_cptr_t?
                var matchedLength: sz_size_t = 0
                stringZillaCheck(
                    sz_utf8_uncased_search_best(
                        hPointer,
                        hLength,
                        &prepared,
                        &match,
                        &matchedLength,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
                if let matchPointer = match {
                    let start = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                    let endPointer = matchPointer.advanced(by: Int(matchedLength))
                    let end = self.stringZillaByteOffset(forByte: endPointer, after: hPointer)
                    result = start ..< end
                }
            }
        }
        return result
    }

    /// Splits the content into UAX-29 words per Unicode TR29, in order. Unlike whitespace
    /// splitting, the words tile the input: every byte belongs to exactly one word.
    /// - Returns: Byte-accurate ranges into the receiver, one per word.
    public func utf8Words() -> [Range<Index>] {
        var ranges: [Range<Index>] = []
        withStringZillaScope { pointer, length in
            let capacity = 64
            var lengths = [sz_size_t](repeating: 0, count: capacity)
            var start: sz_size_t = 0
            while start < length {
                var count: sz_size_t = 0
                stringZillaCheck(
                    sz_utf8_wordbreaks_best(
                        pointer.advanced(by: Int(start)),
                        length - start,
                        &lengths,
                        sz_size_t(capacity),
                        &count,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
                for wordLength in lengths[..<Int(count)] {
                    let lower = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(start)), after: pointer)
                    start += wordLength
                    let upper = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(start)), after: pointer)
                    ranges.append(lower ..< upper)
                }
            }
        }
        return ranges
    }

    /// Splits the content on UTF-8 newline delimiters: the 7 line-break characters plus
    /// a CRLF pair.
    ///
    /// The delimiters partition the text into the N+1 _gaps_ between them, so a string with N
    /// newlines yields N+1 segments. By default empty segments are kept (`skipEmpty: false`), so
    /// `"a\n\nb\n".utf8Lines()` → `["a", "", "b", ""]`.
    ///
    /// - Note: This differs from the Swift standard library's
    ///   `split(omittingEmptySubsequences: true)`, which drops empty subsequences by default.
    ///   Pass `skipEmpty: true` for that behavior.
    /// - Parameter skipEmpty: When `true`, zero-length segments are omitted, defaulting to `false`.
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    public func utf8Lines(skipEmpty: Bool = false) -> [Range<Index>] {
        return utf8Split(skipEmpty: skipEmpty, onNewlines: true)
    }

    /// Splits the content on UTF-8 whitespace delimiters: all 25 Unicode `White_Space` characters.
    ///
    /// The delimiters partition the text into the N+1 _gaps_ between them, so a string with N
    /// whitespace characters yields N+1 segments. By default, with `skipEmpty: false`, empty
    /// segments are kept. Pass `skipEmpty: true` to drop runs of whitespace as separators, as in
    /// `"  hi  ".utf8Tokens(skipEmpty: true)` → `["hi"]`.
    ///
    /// - Note: This differs from the Swift standard library's
    ///   `split(omittingEmptySubsequences: true)`, which drops empty subsequences by default.
    ///   Pass `skipEmpty: true` for that behavior.
    /// - Parameter skipEmpty: When `true`, zero-length segments are omitted, defaulting to `false`.
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    public func utf8Tokens(skipEmpty: Bool = false) -> [Range<Index>] {
        return utf8Split(skipEmpty: skipEmpty, onNewlines: false)
    }

    /// Shared driver for delimiter-based UTF-8 splitting (`utf8Lines` / `utf8Tokens`).
    ///
    /// Batches delimiters through the tokenizer kernel, like `utf8Words()` batches words, but
    /// whereas words _tile_ the input, the delimiters here are discarded and the _gaps_ between
    /// them become the segments. A full batch resumes right after its last delimiter, and the gap
    /// after the last delimiter, possibly empty, closes the split, so N delimiters always produce
    /// N+1 segments.
    ///
    /// - Parameters:
    ///   - skipEmpty: When `true`, zero-length segments are omitted.
    ///   - onNewlines: When `true`, calls `sz_utf8_newlines_best`, else `sz_utf8_whitespaces_best`.
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    private func utf8Split(skipEmpty: Bool, onNewlines: Bool) -> [Range<Index>] {
        let tokenize = onNewlines ? sz_utf8_newlines_best : sz_utf8_whitespaces_best
        var ranges: [Range<Index>] = []
        withStringZillaScope { pointer, length in
            let capacity = 64
            var offsets = [sz_size_t](repeating: 0, count: capacity)
            var lengths = [sz_size_t](repeating: 0, count: capacity)
            var gapStart: sz_size_t = 0

            func appendSegment(until gapEnd: sz_size_t) {
                if skipEmpty && gapEnd == gapStart { return }
                let lower = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(gapStart)), after: pointer)
                let upper = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(gapEnd)), after: pointer)
                ranges.append(lower ..< upper)
            }

            var count: sz_size_t = 0
            repeat {
                let scanStart = gapStart
                var consumed: sz_size_t = 0
                stringZillaCheck(
                    tokenize(
                        pointer.advanced(by: Int(scanStart)),
                        length - scanStart,
                        &offsets,
                        &lengths,
                        sz_size_t(capacity),
                        &count,
                        &consumed,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
                for delimiter in 0 ..< Int(count) {
                    appendSegment(until: scanStart + offsets[delimiter])
                    gapStart = scanStart + offsets[delimiter] + lengths[delimiter]
                }
            } while count == capacity
            appendSegment(until: length)
        }
        return ranges
    }

    /// Lexicographic byte-order comparison, SIMD-accelerated via `sz_order_best`.
    /// - Parameter other: The string to compare against.
    /// - Returns: `.ascending`, `.equal`, or `.descending`.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func compare<S: StringZillaViewable>(_ other: S) -> StringZillaOrdering {
        var ordering = sz_equal_k
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                stringZillaCheck(
                    sz_order_best(aPointer, aLength, bPointer, bLength, &ordering, Device.cpuEnabled.native, nil)
                )
            }
        }
        if ordering == sz_less_k { return .ascending }
        if ordering == sz_greater_k { return .descending }
        return .equal
    }

    /// Uncased comparison using full Unicode case folding, via `sz_utf8_uncased_order_best`.
    /// - Parameter other: The string to compare against.
    /// - Returns: `.ascending`, `.equal`, or `.descending`.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func utf8UncasedOrder<S: StringZillaViewable>(_ other: S) -> StringZillaOrdering {
        var ordering = sz_equal_k
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                stringZillaCheck(
                    sz_utf8_uncased_order_best(
                        aPointer,
                        aLength,
                        bPointer,
                        bLength,
                        &ordering,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
            }
        }
        if ordering == sz_less_k { return .ascending }
        if ordering == sz_greater_k { return .descending }
        return .equal
    }

    /// Byte-level equality, SIMD-accelerated by `sz_equal_best`; unequal lengths never match.
    /// - Parameter other: The string to compare against.
    /// - Returns: `true` if the byte contents are identical.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func equals<S: StringZillaViewable>(_ other: S) -> Bool {
        var equal = sz_false_k
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                guard aLength == bLength else { return }
                stringZillaCheck(sz_equal_best(aPointer, bPointer, aLength, &equal, Device.cpuEnabled.native, nil))
            }
        }
        return equal == sz_true_k
    }
}

/// Uncased search pattern for UTF-8 strings, prepared once for many searches.
/// It is immutable once built, so one needle can search from many threads at once.
public final class Utf8UncasedNeedle: @unchecked Sendable {
    private let bytes: UnsafeMutablePointer<CChar>  // The needle copy that `prepared` points into.
    private let prepared: sz_utf8_uncased_needle_t

    public init<S: StringZillaViewable>(_ needle: S) {
        var prepared = sz_utf8_uncased_needle_t()
        bytes = needle.withStringZillaScope { pointer, length in
            let copy = UnsafeMutablePointer<CChar>.allocate(capacity: max(Int(length), 1))
            copy.initialize(from: pointer, count: Int(length))
            stringZillaCheck(
                sz_utf8_uncased_needle_init_best(copy, length, &prepared, Device.cpuEnabled.native, nil)
            )
            return copy
        }
        self.prepared = prepared
    }

    deinit { bytes.deallocate() }

    /// Finds the first uncased match of the needle, whose byte length folding may change.
    public func findFirst<S: StringZillaViewable>(in haystack: S) -> Range<S.Index>? {
        var result: Range<S.Index>?
        haystack.withStringZillaScope { hPointer, hLength in
            withUnsafePointer(to: prepared) { needle in
                var match: sz_cptr_t?
                var matchedLength: sz_size_t = 0
                stringZillaCheck(
                    sz_utf8_uncased_search_best(
                        hPointer,
                        hLength,
                        needle,
                        &match,
                        &matchedLength,
                        Device.cpuEnabled.native,
                        nil
                    )
                )
                if let matchPointer = match {
                    let start = haystack.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                    let endPointer = matchPointer.advanced(by: Int(matchedLength))
                    let end = haystack.stringZillaByteOffset(forByte: endPointer, after: hPointer)
                    result = start ..< end
                }
            }
        }

        return result
    }
}

/// A progressive hasher for computing StringZilla hashes incrementally. Use this class when you
/// need to hash data that arrives in chunks or when building up a hash over time.
public class StringZillaHasher {
    private var state: sz_hash_state_t

    /// Creates a new hasher with the specified seed.
    /// - Parameter seed: The seed value for the hash function, defaulting to 0.
    public init(seed: UInt64 = 0) {
        state = sz_hash_state_t()
        stringZillaCheck(sz_hash_state_init_best(&state, sz_u64_t(seed), Device.cpuEnabled.native, nil))
    }

    deinit {
        // StringZilla hash state doesn't require explicit cleanup
    }

    /// Updates the hash state with additional string content.
    /// - Parameter content: The string content to add to the hash.
    /// - Returns: Self for method chaining.
    @discardableResult
    public func update<S: StringZillaViewable>(_ content: S) -> StringZillaHasher {
        content.withStringZillaScope { pointer, length in
            stringZillaCheck(sz_hash_state_update_best(&state, pointer, length, Device.cpuEnabled.native, nil))
        }
        return self
    }

    /// Finalizes the hash computation and returns the result.
    /// - Returns: The computed 64-bit hash value.
    /// - Note: This is a non-consuming operation and can be called multiple times.
    public func finalize() -> UInt64 {
        var hash = sz_u64_t()
        stringZillaCheck(sz_hash_state_digest_best(&state, &hash, Device.cpuEnabled.native, nil))
        return UInt64(hash)
    }

    /// Alias for `finalize()`.
    public func digest() -> UInt64 { return finalize() }

    /// Resets the hasher to its initial state.
    /// - Parameter seed: New seed value; the original seed is not retained, so omitting this
    ///   re-seeds with 0.
    public func reset(seed: UInt64? = nil) {
        let newSeed = seed ?? 0  // Default to 0 if no seed provided
        stringZillaCheck(sz_hash_state_init_best(&state, sz_u64_t(newSeed), Device.cpuEnabled.native, nil))
    }
}

/// A progressive SHA-256 hasher for computing cryptographic checksums incrementally. Use this class
/// when you need to hash data that arrives in chunks or when building up a hash over time.
public class StringZillaSha256 {
    private var state: sz_sha256_state_t

    /// Creates a new SHA-256 hasher.
    public init() {
        state = sz_sha256_state_t()
        stringZillaCheck(sz_sha256_state_init_best(&state, Device.cpuEnabled.native, nil))
    }

    deinit {
        // StringZilla SHA-256 state doesn't require explicit cleanup
    }

    /// Updates the hash state with additional data.
    /// - Parameter content: The data to add to the hash.
    /// - Returns: Self for method chaining.
    @discardableResult
    public func update<S: StringZillaViewable>(_ content: S) -> StringZillaSha256 {
        content.withStringZillaScope { pointer, length in
            stringZillaCheck(sz_sha256_state_update_best(&state, pointer, length, Device.cpuEnabled.native, nil))
        }
        return self
    }

    /// Updates the hash state with raw byte data.
    /// - Parameter data: The byte data to add to the hash.
    /// - Returns: Self for method chaining.
    @discardableResult
    public func update(_ data: [UInt8]) -> StringZillaSha256 {
        data.withUnsafeBufferPointer { bufferPointer in
            withStringZillaBuffer(bufferPointer) { pointer, length in
                stringZillaCheck(sz_sha256_state_update_best(&state, pointer, length, Device.cpuEnabled.native, nil))
            }
        }
        return self
    }

    /// Finalizes the hash computation and returns the result as a 32-byte array.
    /// - Returns: The computed SHA-256 digest.
    /// - Note: This is a non-consuming operation and can be called multiple times.
    public func finalize() -> [UInt8] {
        var digest = [UInt8](repeating: 0, count: 32)
        digest.withUnsafeMutableBufferPointer { bufferPointer in
            stringZillaCheck(
                sz_sha256_state_digest_best(&state, bufferPointer.baseAddress!, Device.cpuEnabled.native, nil)
            )
        }
        return digest
    }

    /// Alias for `finalize()`.
    public func digest() -> [UInt8] { return finalize() }

    /// Returns the current SHA-256 hash as a lowercase hexadecimal string.
    /// - Returns: A 64-character hex string.
    public func hexdigest() -> String {
        let digest = self.digest()
        let hexDigits = "0123456789abcdef"
        var result = ""
        result.reserveCapacity(digest.count * 2)
        for byte in digest {
            result.append(hexDigits[hexDigits.index(hexDigits.startIndex, offsetBy: Int(byte >> 4))])
            result.append(hexDigits[hexDigits.index(hexDigits.startIndex, offsetBy: Int(byte & 0x0F))])
        }
        return result
    }

    /// Resets the hasher to its initial state.
    public func reset() {
        stringZillaCheck(sz_sha256_state_init_best(&state, Device.cpuEnabled.native, nil))
    }
}

extension StringZillaViewable {
    /// Computes the SHA-256 cryptographic hash of the content.
    /// - Returns: A 32-byte array containing the SHA-256 digest.
    public func sha256() -> [UInt8] {
        var state = sz_sha256_state_t()
        stringZillaCheck(sz_sha256_state_init_best(&state, Device.cpuEnabled.native, nil))
        withStringZillaScope { pointer, length in
            stringZillaCheck(sz_sha256_state_update_best(&state, pointer, length, Device.cpuEnabled.native, nil))
        }
        var digest = [UInt8](repeating: 0, count: 32)
        digest.withUnsafeMutableBufferPointer { bufferPointer in
            stringZillaCheck(
                sz_sha256_state_digest_best(&state, bufferPointer.baseAddress!, Device.cpuEnabled.native, nil)
            )
        }
        return digest
    }
}

// MARK: - Capabilities and Devices

/// A set of capabilities of a CPU or a GPU, as a ``Device`` reports them.
public struct Capabilities: OptionSet, Sendable, CustomStringConvertible {
    public let rawValue: UInt64
    public init(rawValue: UInt64) { self.rawValue = rawValue }

    /// The C API's mask: `UInt` on Linux, where `sz_capability_t` is `unsigned long`.
    var native: sz_capability_t { sz_capability_t(rawValue) }

    public static let serial = Capabilities(rawValue: 1 << 0)
    public static let westmere = Capabilities(rawValue: 1 << 1)
    public static let goldmont = Capabilities(rawValue: 1 << 2)
    public static let haswell = Capabilities(rawValue: 1 << 3)
    public static let skylake = Capabilities(rawValue: 1 << 4)
    public static let icelake = Capabilities(rawValue: 1 << 5)
    public static let neon = Capabilities(rawValue: 1 << 6)
    public static let neonAes = Capabilities(rawValue: 1 << 7)
    public static let neonSha = Capabilities(rawValue: 1 << 8)
    public static let sve = Capabilities(rawValue: 1 << 9)
    public static let sve2 = Capabilities(rawValue: 1 << 10)
    public static let sve2Aes = Capabilities(rawValue: 1 << 11)
    public static let rvv = Capabilities(rawValue: 1 << 12)
    public static let rvvCrypto = Capabilities(rawValue: 1 << 13)
    public static let v128 = Capabilities(rawValue: 1 << 14)
    public static let v128Relaxed = Capabilities(rawValue: 1 << 15)
    public static let loongsonAsx = Capabilities(rawValue: 1 << 16)
    public static let powerVsx = Capabilities(rawValue: 1 << 17)

    public static let cuda = Capabilities(rawValue: 1 << 48)
    public static let rocm = Capabilities(rawValue: 1 << 56)
    public static let metal = Capabilities(rawValue: 1 << 60)

    /// Every CPU capability, the bits below the first GPU vendor's.
    public static let cpus = Capabilities(rawValue: (1 << 48) - 1)
    /// Every GPU capability.
    public static let devices: Capabilities = [.cuda, .rocm, .metal]
    /// Every capability.
    public static let any = Capabilities(rawValue: .max)

    /// The capability names, comma-separated, like "serial,neon".
    public var description: String {
        String(unsafeUninitializedCapacity: Int(STRINGZILLA_CAPABILITIES_NAME_CAPACITY)) { names in
            names.withMemoryRebound(to: CChar.self) {
                Int(sz_capabilities_name(native, $0.baseAddress, sz_size_t($0.count)))
            }
        }
    }
}

/// Which runtime a device belongs to, as the `sz_<kind>_*` C functions name it.
public enum DeviceKind: Sendable {
    case cpu, cuda, rocm, metal
}

/// Why a ``Device`` query failed, as `sz_status_name` spells the C status.
public struct DeviceError: Error, CustomStringConvertible {
    public let description: String
    init(_ status: sz_status_t) { description = String(cString: sz_status_name(status)) }
}

/// One device StringZilla knows: the host CPU, or a GPU by its runtime's own ordinal, the one
/// `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list.
///
/// Every call of this module runs on the CPU and dispatches over its ``capabilitiesEnabled``, so a
/// GPU device only reports its capabilities here. Prefer ``capabilitiesEnabled`` unless you
/// specifically mean one of the raw axes: ``capabilitiesDetected`` describes the device and says
/// nothing about whether a kernel was compiled into this binary.
public struct Device: Sendable, Equatable {
    public let kind: DeviceKind
    public let ordinal: Int

    /// The host CPU, which every build has.
    public static let cpu = Device(kind: .cpu, unchecked: 0)

    /// The CPU mask every call passes; ``capabilitiesEnable(_:)`` writes it unsynchronized, so
    /// narrow before threads start.
    nonisolated(unsafe) static var cpuEnabled: Capabilities = {
        var mask = Capabilities.serial.native
        _ = sz_cpu_capabilities_enabled(&mask)
        return Capabilities(rawValue: UInt64(mask))
    }()

    private init(kind: DeviceKind, unchecked ordinal: Int) {
        self.kind = kind
        self.ordinal = ordinal
    }

    /// Device `ordinal` of `kind`.
    /// - Throws: ``DeviceError`` past the last device of `kind`.
    public init(kind: DeviceKind, ordinal: Int) throws {
        guard ordinal >= 0, ordinal < (try Device.count(kind)) else { throw DeviceError(sz_missing_gpu_k) }
        self.init(kind: kind, unchecked: ordinal)
    }

    /// How many devices of `kind` the process sees: one CPU, or the GPUs its runtime counts.
    /// - Throws: ``DeviceError`` without a GPU of `kind`.
    public static func count(_ kind: DeviceKind) throws -> Int {
        var count: sz_size_t = 1
        switch kind {
        case .cpu: break
        case .cuda: try check(sz_cuda_count_devices(&count))
        case .rocm: try check(sz_rocm_count_devices(&count))
        case .metal: try check(sz_metal_count_devices(&count))
        }
        return Int(count)
    }

    /// What this device runs, whether or not this binary holds kernels for it.
    public var capabilitiesDetected: Capabilities {
        get throws {
            var mask: sz_capability_t = 0
            let device = sz_size_t(ordinal)
            switch kind {
            case .cpu: try Device.check(sz_cpu_capabilities_detected(&mask))
            case .cuda: try Device.check(sz_cuda_capabilities_detected(device, &mask))
            case .rocm: try Device.check(sz_rocm_capabilities_detected(device, &mask))
            case .metal: try Device.check(sz_metal_capabilities_detected(device, &mask))
            }
            return Capabilities(rawValue: UInt64(mask))
        }
    }

    /// What this binary holds kernels for on devices of this kind, whether or not this one runs.
    public var capabilitiesCompiled: Capabilities {
        var mask: sz_capability_t = 0
        switch kind {
        case .cpu: _ = sz_cpu_capabilities_compiled(&mask)
        case .cuda: _ = sz_cuda_capabilities_compiled(&mask)
        case .rocm: _ = sz_rocm_capabilities_compiled(&mask)
        case .metal: _ = sz_metal_capabilities_compiled(&mask)
        }
        return Capabilities(rawValue: UInt64(mask))
    }

    /// What this device's calls pass: ``capabilitiesDetected`` and ``capabilitiesCompiled`` at
    /// once. On the CPU it is what every call of this module passes, narrowed by
    /// ``capabilitiesEnable(_:)``, and always contains ``Capabilities/serial``.
    public var capabilitiesEnabled: Capabilities {
        get throws {
            var mask: sz_capability_t = 0
            let device = sz_size_t(ordinal)
            switch kind {
            case .cpu: return Device.cpuEnabled
            case .cuda: try Device.check(sz_cuda_capabilities_enabled(device, &mask))
            case .rocm: try Device.check(sz_rocm_capabilities_enabled(device, &mask))
            case .metal: try Device.check(sz_metal_capabilities_enabled(device, &mask))
            }
            return Capabilities(rawValue: UInt64(mask))
        }
    }

    /// Makes `wanted` the CPU's ``capabilitiesEnabled`` set, clamped to what it detects and this
    /// binary compiled and keeping ``Capabilities/serial``.
    /// - Returns: The set that took effect.
    /// - Throws: ``DeviceError`` on a GPU, which keeps no such set.
    @discardableResult
    public func capabilitiesEnable(_ wanted: Capabilities) throws -> Capabilities {
        guard kind == .cpu else { throw DeviceError(sz_missing_kernel_k) }
        var mask = Capabilities.serial.native
        _ = sz_cpu_capabilities_enabled(&mask)
        Device.cpuEnabled = wanted.intersection(Capabilities(rawValue: UInt64(mask))).union(.serial)
        return Device.cpuEnabled
    }

    /// Configures the current thread for `capabilities`, usually ``capabilitiesEnabled``. Call it
    /// once on every thread that runs kernels.
    /// - Throws: ``DeviceError`` on a GPU, which has no thread state to configure.
    public func configureThread(_ capabilities: Capabilities) throws {
        guard kind == .cpu else { throw DeviceError(sz_missing_kernel_k) }
        try Device.check(sz_cpu_configure_thread(capabilities.native))
    }

    private static func check(_ status: sz_status_t) throws {
        guard status == sz_success_k else { throw DeviceError(status) }
    }
}
