// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "StringZilla",
    products: [
        .library(name: "StringZilla", targets: ["StringZillaC", "StringZilla"])
    ],
    targets: [
        .target(
            name: "StringZillaC",
            path: "include/stringzilla",
            sources: ["empty.c"],
            publicHeadersPath: "."
        ),
        .target(
            name: "StringZilla",
            dependencies: ["StringZillaC"],
            path: "swift",
            exclude: ["Test.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZilla"],
            path: "swift",
            sources: ["Test.swift"]
        )
    ],
    cLanguageStandard: CLanguageStandard.c99
)
