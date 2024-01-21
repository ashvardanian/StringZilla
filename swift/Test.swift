//
//  File.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//

import Foundation
import StringZilla
import XCTest

@available(iOS 13, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
class Test: XCTestCase {
    func testUnit() throws {
        let str = "Hello, playground, playground, playground"
        assert(str.find("play") == 7)
    }
}
