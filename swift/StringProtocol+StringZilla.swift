//
//  StringProtocol+StringZilla.swift
//
//  Created by Ash Vardanian on 18/1/24.
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

import StringZillaC

// We need to link the standard libraries.
#if os(Linux)
    import Glibc
#else
    import Darwin.C
#endif

/// Result of a three-way string comparison.
///
/// Mirrors `Foundation.ComparisonResult` but is defined here so the package stays Foundation-free and usable
/// on Linux and embedded targets. The cases match the `sz_order` / `sz_utf8_uncased_order` verdicts.
public enum StringZillaOrdering: Sendable {
    case ascending  // The receiver sorts before the argument.
    case equal  // The two compare as equal.
    case descending  // The receiver sorts after the argument.
}

/// Unicode normalization form selector, mirroring `sz_normal_form_t`.
public enum StringZillaNormalizationForm: RawRepresentable, Sendable {
    case nfd   // Canonical decomposition.
    case nfc   // Canonical decomposition + canonical composition.
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

@usableFromInline
enum StringZillaError: Error {
    case contiguousStorageUnavailable
    case memoryAllocationFailed

    var localizedDescription: String {
        switch self {
        case .contiguousStorageUnavailable:
            return "Contiguous storage for the sequence is unavailable."
        case .memoryAllocationFailed:
            return "Memory allocation failed."
        }
    }
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
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `szScope`.
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

    /// Executes a closure with a pointer to the UTF8View's contiguous storage of single-byte elements (UTF-8 code units).
    /// - Parameters:
    ///   - body: A closure that takes a pointer to the contiguous storage and its size.
    /// - Throws: An error if the storage is not contiguous.
    @_transparent
    public func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R {
        return try withContiguousStorageIfAvailable { bufferPointer -> R in
            let cLength = sz_size_t(bufferPointer.count)
            let cString = UnsafeRawPointer(bufferPointer.baseAddress!).assumingMemoryBound(to: CChar.self)
            return try body(cString, cLength)
        }
            ?? {
                throw StringZillaError.contiguousStorageUnavailable
            }()
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `szScope`.
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

    /// Executes a closure with a pointer to the UTF8View's contiguous storage of single-byte elements (UTF-8 code units).
    /// - Parameters:
    ///   - body: A closure that takes a pointer to the contiguous storage and its size.
    /// - Throws: An error if the storage is not contiguous.
    public func withStringZillaScope<R>(_ body: (sz_cptr_t, sz_size_t) throws -> R) rethrows -> R {
        return try withContiguousStorageIfAvailable { bufferPointer -> R in
            let cLength = sz_size_t(bufferPointer.count)
            let cString = UnsafeRawPointer(bufferPointer.baseAddress!).assumingMemoryBound(to: CChar.self)
            return try body(cString, cLength)
        }
            ?? {
                throw StringZillaError.contiguousStorageUnavailable
            }()
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `szScope`.
    /// - Returns: The calculated index offset.
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t)
        -> Index
    {
        return index(startIndex, offsetBy: bytePointer - startPointer)
    }
}

extension StringZillaViewable {
    /// Computes a 64-bit hash of the string content using StringZilla's fast hash algorithm.
    /// - Parameter seed: Optional seed value for the hash function (default: 0).
    /// - Returns: A 64-bit unsigned integer hash value.
    public func hash(seed: UInt64 = 0) -> UInt64 {
        return withStringZillaScope { pointer, length in
            sz_hash(pointer, length, seed)
        }
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
                if let matchPointer = sz_find(hPointer, hLength, nPointer, nLength) {
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
                if let matchPointer = sz_rfind(hPointer, hLength, nPointer, nLength) {
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
        withStringZillaScope { hPointer, hLength in
            characters.withStringZillaScope { nPointer, nLength in
                if let matchPointer = sz_find_byte_from(hPointer, hLength, nPointer, nLength) {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
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
        withStringZillaScope { hPointer, hLength in
            characters.withStringZillaScope { nPointer, nLength in
                if let matchPointer = sz_rfind_byte_from(hPointer, hLength, nPointer, nLength) {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the first occurrence of a character outside of the the given character-set within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findFirst<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
        var result: Index?
        withStringZillaScope { hPointer, hLength in
            characters.withStringZillaScope { nPointer, nLength in
                if let matchPointer = sz_find_byte_not_from(hPointer, hLength, nPointer, nLength) {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the last occurrence of a character outside of the the given character-set within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func findLast<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
        var result: Index?
        withStringZillaScope { hPointer, hLength in
            characters.withStringZillaScope { nPointer, nLength in
                if let matchPointer = sz_rfind_byte_not_from(hPointer, hLength, nPointer, nLength) {
                    result = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Applies full Unicode case folding to the content's UTF-8 bytes.
    /// The returned bytes are UTF-8 and may be longer than the input (e.g., "ß" -> "ss").
    public func utf8UncasedFoldedBytes() -> [UInt8] {
        var folded: [UInt8] = []
        withStringZillaScope { pointer, length in
            if length == 0 {
                folded = []
                return
            }
            let capacity = Int(length) * 3
            var destination = [UInt8](repeating: 0, count: capacity)
            let outLen: sz_size_t = destination.withUnsafeMutableBufferPointer { bufferPointer in
                sz_utf8_uncased_fold(pointer, length, bufferPointer.baseAddress)
            }
            let actual = Int(outLen)
            if actual < destination.count { destination.removeLast(destination.count - actual) }
            folded = destination
        }
        return folded
    }

    /// Produces the UTF-8 bytes of the receiver after applying a Unicode normalization form.
    /// The output may be longer than the input; the worst-case expansion is 18× per source byte (NFKD).
    /// - Parameter form: The normalization form to apply (default: `.nfc`).
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
            let outLen: sz_size_t = destination.withUnsafeMutableBufferPointer { bufferPointer in
                sz_utf8_norm(pointer, length, form.rawValue, bufferPointer.baseAddress)
            }
            let actual = Int(outLen)
            if actual < destination.count { destination.removeLast(destination.count - actual) }
            normalized = destination
        }
        return normalized
    }

    /// Returns the index of the first byte violating the given normalization form, or `nil` if already normalized.
    /// - Parameter form: The normalization form to test.
    /// - Returns: The `Index` of the first non-conforming byte, or `nil` if the content is already in @p form.
    public func utf8NormalizationViolation(_ form: StringZillaNormalizationForm) -> Index? {
        var result: Index?
        withStringZillaScope { pointer, length in
            if let violationPointer = sz_utf8_find_denormalized(pointer, length, form.rawValue) {
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
                var metadata = sz_utf8_uncased_needle_metadata_t()
                var matchedLength: sz_size_t = 0
                if let matchPointer = sz_utf8_uncased_search(
                    hPointer,
                    hLength,
                    nPointer,
                    nLength,
                    &metadata,
                    &matchedLength
                ) {
                    let start = self.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                    let endPointer = matchPointer.advanced(by: Int(matchedLength))
                    let end = self.stringZillaByteOffset(forByte: endPointer, after: hPointer)
                    result = start..<end
                }
            }
        }
        return result
    }

    /// Splits the content into UAX-29 words (Unicode TR29), in order.
    /// Unlike whitespace splitting, the words tile the input: every byte belongs to exactly one word.
    /// - Returns: Byte-accurate ranges into the receiver, one per word.
    public func utf8Words() -> [Range<Index>] {
        var ranges: [Range<Index>] = []
        withStringZillaScope { pointer, length in
            var cursor: sz_size_t = 0
            while cursor < length {
                var wordStart: sz_size_t = 0, wordLength: sz_size_t = 0, consumed: sz_size_t = 0
                let count = sz_utf8_words(
                    pointer.advanced(by: Int(cursor)), length - cursor, &wordStart, &wordLength, 1, &consumed)
                if count == 0 { break }
                let begin = cursor + wordStart // The first word of the suffix starts at offset 0.
                let end = begin + wordLength
                let lo = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(begin)), after: pointer)
                let hi = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(end)), after: pointer)
                ranges.append(lo..<hi)
                cursor = end
            }
        }
        return ranges
    }

    /// Splits the content on UTF-8 newline delimiters (the 7 line-break characters plus a CRLF pair).
    ///
    /// The delimiters partition the text into the N+1 *gaps* between them, so a string with N newlines
    /// yields N+1 segments. By default empty segments are kept (`skipEmpty: false`), which mirrors the
    /// cross-language KEEP policy and matches `"a\n\nb\n".utf8Lines()` -> `["a", "", "b", ""]`.
    ///
    /// - Note: This differs from the Swift standard library's `split(omittingEmptySubsequences: true)`,
    ///   which drops empty subsequences by default. Pass `skipEmpty: true` for that behavior.
    /// - Parameter skipEmpty: When `true`, zero-length segments are omitted (default: `false`).
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    public func utf8Lines(skipEmpty: Bool = false) -> [Range<Index>] {
        return utf8Split(skipEmpty: skipEmpty, onNewlines: true)
    }

    /// Splits the content on UTF-8 whitespace delimiters (all 25 Unicode `White_Space` characters).
    ///
    /// The delimiters partition the text into the N+1 *gaps* between them, so a string with N whitespace
    /// characters yields N+1 segments. By default empty segments are kept (`skipEmpty: false`), which mirrors
    /// the cross-language KEEP policy. Pass `skipEmpty: true` to drop runs of whitespace as separators, e.g.
    /// `"  hi  ".utf8Tokens(skipEmpty: true)` -> `["hi"]`.
    ///
    /// - Note: This differs from the Swift standard library's `split(omittingEmptySubsequences: true)`,
    ///   which drops empty subsequences by default. Pass `skipEmpty: true` for that behavior.
    /// - Parameter skipEmpty: When `true`, zero-length segments are omitted (default: `false`).
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    public func utf8Tokens(skipEmpty: Bool = false) -> [Range<Index>] {
        return utf8Split(skipEmpty: skipEmpty, onNewlines: false)
    }

    /// Shared driver for delimiter-based UTF-8 splitting (`utf8Lines` / `utf8Tokens`).
    ///
    /// Buffers delimiter boundaries through the multistep FFI kernel (just like `utf8Words()`), but whereas
    /// words *tile* the input, the delimiters here are discarded and the *gaps* between them become the
    /// segments: delimiter `d` spans `[start, start + length)`, and the segment preceding it runs from the
    /// previous delimiter's end up to this delimiter's start. When the kernel reports it consumed the whole
    /// remaining region, the trailing segment after the last delimiter (possibly empty) is appended too, so
    /// N delimiters always produce N+1 segments.
    ///
    /// - Parameters:
    ///   - skipEmpty: When `true`, zero-length segments are omitted.
    ///   - onNewlines: When `true`, drives `sz_utf8_newlines`; otherwise `sz_utf8_whitespaces`.
    /// - Returns: Byte-accurate ranges into the receiver, one per segment.
    private func utf8Split(skipEmpty: Bool, onNewlines: Bool) -> [Range<Index>] {
        var ranges: [Range<Index>] = []
        withStringZillaScope { pointer, length in
            // Buffer a handful of delimiters per FFI call, then transform them into segment gaps.
            let steps = Int(sz_iterators_default_steps_k)
            var offsets = [sz_size_t](repeating: 0, count: steps)
            var lengths = [sz_size_t](repeating: 0, count: steps)
            var suffix: sz_size_t = 0 // Byte offset of the not-yet-segmented suffix within `pointer`.

            // Emits one segment `[begin, end)` (offsets relative to `pointer`), honoring `skipEmpty`.
            func appendSegment(_ begin: sz_size_t, _ end: sz_size_t) {
                if skipEmpty && end == begin { return }
                let lo = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(begin)), after: pointer)
                let hi = self.stringZillaByteOffset(forByte: pointer.advanced(by: Int(end)), after: pointer)
                ranges.append(lo..<hi)
            }

            while suffix <= length {
                let region = length - suffix
                var consumed: sz_size_t = 0
                let delimiters = offsets.withUnsafeMutableBufferPointer { offsetsBuffer in
                    lengths.withUnsafeMutableBufferPointer { lengthsBuffer in
                        onNewlines
                            ? sz_utf8_newlines(
                                pointer.advanced(by: Int(suffix)), region, offsetsBuffer.baseAddress,
                                lengthsBuffer.baseAddress, sz_size_t(steps), &consumed)
                            : sz_utf8_whitespaces(
                                pointer.advanced(by: Int(suffix)), region, offsetsBuffer.baseAddress,
                                lengthsBuffer.baseAddress, sz_size_t(steps), &consumed)
                    }
                }
                // Each delimiter's gap (the segment before it) becomes one output range; offsets are
                // relative to `pointer.advanced(by: suffix)`, so re-base them onto `pointer` via `suffix`.
                var previousEnd: sz_size_t = 0
                for delimiter in 0..<Int(delimiters) {
                    let delimiterStart = offsets[delimiter]
                    let delimiterLength = lengths[delimiter]
                    appendSegment(suffix + previousEnd, suffix + delimiterStart)
                    previousEnd = delimiterStart + delimiterLength
                }
                if consumed == region {
                    // Reached end-of-text: append the trailing segment after the last delimiter, then stop.
                    appendSegment(suffix + previousEnd, suffix + region)
                    break
                }
                if consumed == 0 { break } // Defensive: never spin in place on a non-advancing batch.
                suffix += consumed
            }
        }
        return ranges
    }

    /// Lexicographic (byte-order) comparison, SIMD-accelerated via `sz_order`.
    /// - Parameter other: The string to compare against.
    /// - Returns: `.ascending`, `.equal`, or `.descending`.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func compare<S: StringZillaViewable>(_ other: S) -> StringZillaOrdering {
        var ordering = sz_equal_k
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                ordering = sz_order(aPointer, aLength, bPointer, bLength)
            }
        }
        if ordering == sz_less_k { return .ascending }
        if ordering == sz_greater_k { return .descending }
        return .equal
    }

    /// Uncased comparison using full Unicode case folding, via `sz_utf8_uncased_order`.
    /// - Parameter other: The string to compare against.
    /// - Returns: `.ascending`, `.equal`, or `.descending`.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func utf8UncasedOrder<S: StringZillaViewable>(_ other: S) -> StringZillaOrdering {
        var ordering = sz_equal_k
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                ordering = sz_utf8_uncased_order(aPointer, aLength, bPointer, bLength)
            }
        }
        if ordering == sz_less_k { return .ascending }
        if ordering == sz_greater_k { return .descending }
        return .equal
    }

    /// Byte-level equality, SIMD-accelerated via `sz_equal` (differing lengths are never equal).
    /// - Parameter other: The string to compare against.
    /// - Returns: `true` if the byte contents are identical.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    public func equals<S: StringZillaViewable>(_ other: S) -> Bool {
        var result = false
        withStringZillaScope { aPointer, aLength in
            other.withStringZillaScope { bPointer, bLength in
                result = aLength == bLength && sz_equal(aPointer, bPointer, aLength) == sz_true_k
            }
        }
        return result
    }
}

/// Pre-compiled uncased search pattern for UTF-8 strings.
/// Caches metadata for efficient repeated searches with the same needle.
public final class Utf8UncasedNeedle {
    private let needleBytes: [UInt8]
    private var metadata: sz_utf8_uncased_needle_metadata_t

    public init<S: StringZillaViewable>(_ needle: S) {
        var bytes: [UInt8] = []
        needle.withStringZillaScope { pointer, length in
            if length == 0 {
                bytes = []
                return
            }
            let start = UnsafeRawPointer(pointer).assumingMemoryBound(to: UInt8.self)
            bytes = Array(UnsafeBufferPointer(start: start, count: Int(length)))
        }
        needleBytes = bytes
        metadata = sz_utf8_uncased_needle_metadata_t()
    }

    /// Note: not safe for concurrent use. The internal metadata is computed lazily and mutated during searches.
    public func findFirst<S: StringZillaViewable>(in haystack: S) -> Range<S.Index>? {
        if needleBytes.isEmpty { return haystack.startIndex..<haystack.startIndex }

        var result: Range<S.Index>?
        haystack.withStringZillaScope { hPointer, hLength in
            needleBytes.withUnsafeBufferPointer { needleBuffer in
                let nPointer = UnsafeRawPointer(needleBuffer.baseAddress!).assumingMemoryBound(to: CChar.self)
                let nLength = sz_size_t(needleBuffer.count)

                var matchedLength: sz_size_t = 0
                if let matchPointer = sz_utf8_uncased_search(
                    hPointer,
                    hLength,
                    nPointer,
                    nLength,
                    &metadata,
                    &matchedLength
                ) {
                    let start = haystack.stringZillaByteOffset(forByte: matchPointer, after: hPointer)
                    let endPointer = matchPointer.advanced(by: Int(matchedLength))
                    let end = haystack.stringZillaByteOffset(forByte: endPointer, after: hPointer)
                    result = start..<end
                }
            }
        }

        return result
    }
}

/// A progressive hasher for computing StringZilla hashes incrementally.
/// Use this class when you need to hash data that arrives in chunks or when building up a hash over time.
public class StringZillaHasher {
    private var state: sz_hash_state_t

    /// Creates a new hasher with the specified seed.
    /// - Parameter seed: The seed value for the hash function (default: 0).
    public init(seed: UInt64 = 0) {
        state = sz_hash_state_t()
        sz_hash_state_init(&state, seed)
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
            sz_hash_state_update(&state, pointer, length)
        }
        return self
    }

    /// Finalizes the hash computation and returns the result.
    /// - Returns: The computed 64-bit hash value.
    /// - Note: This is a non-consuming operation and can be called multiple times.
    public func finalize() -> UInt64 {
        return sz_hash_state_digest(&state)
    }

    /// Alias for `finalize()` to match other bindings.
    public func digest() -> UInt64 { return finalize() }

    /// Resets the hasher to its initial state with the same seed.
    /// - Parameter seed: Optional new seed value (if nil, uses the original seed).
    public func reset(seed: UInt64? = nil) {
        let newSeed = seed ?? 0  // Default to 0 if no seed provided
        sz_hash_state_init(&state, newSeed)
    }
}

/// A progressive SHA-256 hasher for computing cryptographic checksums incrementally.
/// Use this class when you need to hash data that arrives in chunks or when building up a hash over time.
public class StringZillaSha256 {
    private var state: sz_sha256_state_t

    /// Creates a new SHA-256 hasher.
    public init() {
        state = sz_sha256_state_t()
        sz_sha256_state_init(&state)
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
            sz_sha256_state_update(&state, pointer, length)
        }
        return self
    }

    /// Updates the hash state with raw byte data.
    /// - Parameter data: The byte data to add to the hash.
    /// - Returns: Self for method chaining.
    @discardableResult
    public func update(_ data: [UInt8]) -> StringZillaSha256 {
        data.withUnsafeBufferPointer { bufferPointer in
            let cString = UnsafeRawPointer(bufferPointer.baseAddress!).assumingMemoryBound(to: CChar.self)
            sz_sha256_state_update(&state, cString, sz_size_t(bufferPointer.count))
        }
        return self
    }

    /// Finalizes the hash computation and returns the result as a 32-byte array.
    /// - Returns: The computed SHA-256 digest.
    /// - Note: This is a non-consuming operation and can be called multiple times.
    public func finalize() -> [UInt8] {
        var digest = [UInt8](repeating: 0, count: 32)
        digest.withUnsafeMutableBufferPointer { bufferPointer in
            sz_sha256_state_digest(&state, bufferPointer.baseAddress!)
        }
        return digest
    }

    /// Alias for `finalize()` to match other bindings.
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
        sz_sha256_state_init(&state)
    }
}

extension StringZillaViewable {
    /// Computes the SHA-256 cryptographic hash of the content.
    /// - Returns: A 32-byte array containing the SHA-256 digest.
    public func sha256() -> [UInt8] {
        var state = sz_sha256_state_t()
        sz_sha256_state_init(&state)
        withStringZillaScope { pointer, length in
            sz_sha256_state_update(&state, pointer, length)
        }
        var digest = [UInt8](repeating: 0, count: 32)
        digest.withUnsafeMutableBufferPointer { bufferPointer in
            sz_sha256_state_digest(&state, bufferPointer.baseAddress!)
        }
        return digest
    }
}
