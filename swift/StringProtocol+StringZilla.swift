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
