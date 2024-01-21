//
//  File.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//

import Foundation
import StringZillaSwift
import XCTest

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
class Test: XCTestCase {
    func testUnit() throws {
        var str = "Hello, playground, playground, playground"
        if let index = str.find("play") {
            let position = str.distance(from: str.startIndex, to: index)
            assert(position == 7)
        } else {
            assert(false, "Failed to find the substring")
        }
        print("StringZilla Swift test passed 🎉")
    }
}
