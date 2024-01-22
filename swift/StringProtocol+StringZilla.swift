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

import Foundation
import StringZillaC

/// Protocol defining a single-byte data type.
protocol SingleByte {}
extension UInt8: SingleByte {}
extension Int8: SingleByte {} // This would match `CChar` as well.

/// Protocol defining the interface for StringZilla-compatible byte-spans.
///  
/// # Discussion:
/// The Swift documentation is extremely vague about the actual memory layout of a String
/// and the cost of obtaining the underlying UTF8 representation or any other raw pointers.
/// https://developer.apple.com/documentation/swift/stringprotocol/withcstring(_:)
/// https://developer.apple.com/documentation/swift/stringprotocol/withcstring(encodedas:_:)
/// https://developer.apple.com/documentation/swift/stringprotocol/data(using:allowlossyconversion:)
protocol StringZillaView {
    associatedtype Index

    /// Executes a closure with a pointer to the string's UTF8 C representation and its length.
    /// - Parameters:
    ///   - body: A closure that takes a pointer to a C string and its length.
    func withCStringZilla(_ body: (sz_cptr_t, sz_size_t) -> Void)

    /// Calculates the offset index for a given byte pointer relative to a start pointer.
    /// - Parameters:
    ///   - bytePointer: A pointer to the byte for which the offset is calculated.
    ///   - startPointer: The starting pointer for the calculation, previously obtained from `withCStringZilla`.
    /// - Returns: The calculated index offset.
    func getOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index
}

extension String: StringZillaView {
    typealias Index = String.Index
    func withCStringZilla(_ body: (sz_cptr_t, sz_size_t) -> Void) {
        let cLength = sz_size_t(self.lengthOfBytes(using: .utf8))
        self.withCString { cString in
            body(cString, cLength)
        }
    }
    
    func getOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index {
        return self.index(self.startIndex, offsetBy: bytePointer - startPointer)
    }
}

extension UnsafeBufferPointer where Element == SingleByte {
    typealias Index = Int
    func withCStringZilla(_ body: (sz_cptr_t, sz_size_t) -> Void) {
        let cLength = sz_size_t(count)
        let cString = UnsafeRawPointer(self.baseAddress!).assumingMemoryBound(to: CChar.self)
        body(cString, cLength)
    }

    func getOffset(forByte bytePointer: sz_cptr_t, after startPointer: sz_cptr_t) -> Index {
        return Int(bytePointer - startPointer)
    }
}

extension StringZillaView {
    
    /// Finds the first occurrence of the specified substring within the receiver.
    /// - Parameter needle: The substring to search for.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findFirst(_ needle: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            needle.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the last occurrence of the specified substring within the receiver.
    /// - Parameter needle: The substring to search for.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findLast(_ needle: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            needle.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find_last(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the first occurrence of the specified character-set members within the receiver.
    /// - Parameter characters: A string-like collection of characters to match.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findFirst(of characters: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            characters.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the last occurrence of the specified character-set members within the receiver.
    /// - Parameter characters: A string-like collection of characters to match.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findLast(of characters: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            characters.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find_last(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the first occurrence of a character outside of the the given character-set within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findFirst(notOf characters: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            characters.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Finds the last occurrence of a character outside of the the given character-set within the receiver.
    /// - Parameter characters: A string-like collection of characters to exclude.
    /// - Returns: The index of the found occurrence, or `nil` if not found.
    public func findLast(notOf characters: any StringZillaView) -> Index? {
        var result: Index?
        withCStringZilla { hPointer, hLength in
            characters.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_find_last(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }

    /// Computes the Levenshtein edit distance between this and another string.
    /// - Parameter other: A string-like collection of characters to exclude.
    /// - Returns: The edit distance, as an unsigned integer.
    /// - Throws: If a memory allocation error has happened.
    public func editDistance(_ other: any StringZillaView) -> UInt {
        var result: Int?
        withCStringZilla { hPointer, hLength in
            other.withCStringZilla { nPointer, nLength in
                if let matchPointer = sz_edit_distance(hPointer, hLength, nPointer, nLength) {
                    result = self.getOffset(forByte: matchPointer, after: hPointer)
                }
            }
        }
        return result
    }
}
