//
//  Test.swift
//
//
//  Created by Ash Vardanian on 18/1/24.
//

import XCTest
@testable import StringZilla

class StringZillaTests: XCTestCase {
    
    var testString: String!
    
    override func setUp() {
        super.setUp()
        testString = "Hello, world! Welcome to StringZilla. 👋"
        XCTAssertEqual(testString.count, 39)
        XCTAssertEqual(testString.utf8.count, 42)
    }
    
    func testFindFirstSubstring() {
        let index = testString.findFirst(substring: "world")!
        XCTAssertEqual(testString[index...], "world! Welcome to StringZilla. 👋")
    }
    
    func testFindLastSubstring() {
        let index = testString.findLast(substring: "o")!
        XCTAssertEqual(testString[index...], "o StringZilla. 👋")
    }
    
    func testFindFirstCharacterFromSet() {
        let index = testString.findFirst(characterFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "ello, world! Welcome to StringZilla. 👋")
    }
    
    func testFindLastCharacterFromSet() {
        let index = testString.findLast(characterFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "a. 👋")
    }
    
    func testFindFirstCharacterNotFromSet() {
        let index = testString.findFirst(characterNotFrom: "aeiou")!
        XCTAssertEqual(testString[index...], "Hello, world! Welcome to StringZilla. 👋")
    }

     func testFindLastCharacterNotFromSet() {
        let index = testString.findLast(characterNotFrom: "aeiou")!
        XCTAssertEqual(testString.distance(from: testString.startIndex, to: index), 38)
        XCTAssertEqual(testString[index...], "👋")
    }
    
    func testEditDistance() {
        let otherString = "Hello, world!"
        let distance = try? testString.editDistance(from: otherString)
        XCTAssertNotNil(distance)
        XCTAssertEqual(distance, 29)
    }
    
    func testFindLastCharacterNotFromSetNoMatch() {
        let index = "aeiou".findLast(characterNotFrom: "aeiou")
        XCTAssertNil(index)
    }
}
