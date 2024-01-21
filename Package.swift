// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "StringZilla",
    products: [
        .library(name: "StringZilla", targets: ["StringZillaC", "StringZillaSwift"])
    ],
    targets: [
        .target(
            name: "StringZillaC",
            path: "include/stringzilla",
            publicHeadersPath: "."
        ),
        .target(
            name: "StringZillaSwift",
            dependencies: ["StringZillaC"],
            path: "swift",
            exclude: ["Test.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZillaSwift"],
            path: "swift",
            sources: ["Test.swift"]
        )
    ],
    cLanguageStandard: CLanguageStandard.c99,
    cxxLanguageStandard: CXXLanguageStandard.cxx14
)
