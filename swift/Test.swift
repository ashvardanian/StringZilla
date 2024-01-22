//
//  Test.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//

import Foundation
import XCTest

import StringZilla

class Test: XCTestCase {
    func testUnit() throws {
        var str = "Hi there! It's nice to meet you! 👋"
        let endOfSentence = str.firstIndex(of: "!")!
        let firstSentence = str[...endOfSentence]
        assert(firstSentence == "Hi there!")

        if let index = str.utf8.find("play".utf8) {
            let position = str.distance(from: str.startIndex, to: index)
            assert(position == 7)
        } else {
            assert(false, "Failed to find the substring")
        }
        print("StringZilla Swift test passed 🎉")
    }
}
