//
//  StringProtocol+StringZilla.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//
//  Reading materials:
//  - String’s ABI and UTF-8. Nov 2018
//    https://forums.swift.org/t/string-s-abi-and-utf-8/17676
//  - Stable pointer into a C string without copying it? Aug 2021
//    https://forums.swift.org/t/stable-pointer-into-a-c-string-without-copying-it/51244/1

import Foundation
import StringZillaC

extension StringProtocol {
    public mutating func find<S: StringProtocol>(_ other: S) -> Index? {
        var selfSubstring = Substring(self)
        var otherSubstring = Substring(other)

        return selfSubstring.withUTF8 { cSelf in
            otherSubstring.withUTF8 { cOther in
                // Get the byte lengths of the substrings
                let selfLength = cSelf.count
                let otherLength = cOther.count

                // Call the C function
                if let result = sz_find(cSelf.baseAddress, sz_size_t(selfLength), cOther.baseAddress, sz_size_t(otherLength)) {
                    // Calculate the index in the original substring
                    let offset = UnsafeRawPointer(result) - UnsafeRawPointer(cSelf.baseAddress!)
                    return self.index(self.startIndex, offsetBy: offset)
                }
                return nil
            }
        }
    }
}
