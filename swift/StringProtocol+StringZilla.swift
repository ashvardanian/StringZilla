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

/// Protocol defining a single-byte data type.
private protocol SingleByte {}

extension UInt8: SingleByte {}
extension Int8: SingleByte {} // This would match `CChar` as well.

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
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index {
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
        } ?? {
            throw StringZillaError.contiguousStorageUnavailable
        }()
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `szScope`.
    /// - Returns: The calculated index offset.
    @_transparent
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index {
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
        } ?? {
            throw StringZillaError.contiguousStorageUnavailable
        }()
    }

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `szScope`.
    /// - Returns: The calculated index offset.
    public func stringZillaByteOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index {
        return index(startIndex, offsetBy: bytePointer - startPointer)
    }
}

public extension StringZillaViewable {
    /// Finds the first occurrence of the specified substring within the receiver.
    /// - Parameter needle: The substring to search for.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    @_specialize(where Self == String, S == String)
    @_specialize(where Self == String.UTF8View, S == String.UTF8View)
    func findFirst<S: StringZillaViewable>(substring needle: S) -> Index? {
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
    func findLast<S: StringZillaViewable>(substring needle: S) -> Index? {
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
    func findFirst<S: StringZillaViewable>(characterFrom characters: S) -> Index? {
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
    func findLast<S: StringZillaViewable>(characterFrom characters: S) -> Index? {
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
    func findFirst<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
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
    func findLast<S: StringZillaViewable>(characterNotFrom characters: S) -> Index? {
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

    func levenshteinDistance<S: StringZillaViewable>(
        from other: S,
        bound: UInt? = nil
    ) throws -> UInt {
        // Prepare a local variable for the result.
        var computedResult: sz_size_t = 0

        // Swift has a ridiculous issue with casting unsigned 64-bit to unsigned 64-bit
        // values which results in "Fatal error: Not enough bits to represent the passed value".
        // Let's just copy the bytes: https://stackoverflow.com/a/68650250/2766161
        let effectiveBound: sz_size_t = bound.map { sz_size_t($0) } ?? _sz_size_max()
        let status = try withStringZillaScope { hPointer, hLength in
            try other.withStringZillaScope { nPointer, nLength in
                // Pass a mutable pointer for the result.
                sz_levenshtein_distance(
                    hPointer,
                    hLength,
                    nPointer,
                    nLength,
                    effectiveBound,
                    nil, // default allocator
                    &computedResult // out-parameter for the computed distance
                )
            }
        }

        // Check the returned status code.
        guard status == sz_success_k else {
            // Map the status code to an appropriate Swift error.
            throw StringZillaError.memoryAllocationFailed
        }

        return UInt(computedResult)
    }
}
