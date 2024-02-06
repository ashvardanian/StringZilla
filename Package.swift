// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "StringZilla",
    products: [
        .library(
            name: "StringZilla",
            targets: ["StringZillaC", "StringZilla"]
        )
    ],
    targets: [
        .target(
            name: "StringZillaC",
            path: "include/stringzilla", // Adjust the path to include your C source files
            sources: ["../../c/lib.c"], // Include the source file here
            publicHeadersPath: ".",
            cSettings: [
                .define("SZ_DYNAMIC_DISPATCH", to: "1"), // Define a macro
                .define("SZ_AVOID_LIBC", to: "0"), // We need `malloc` from LibC
                .define("SZ_DEBUG", to: "0"), // We don't need any extra assertions in the C layer
                .headerSearchPath("include/stringzilla"), // Specify header search paths
                .unsafeFlags(["-Wall"]) // Use with caution: specify custom compiler flags
            ]
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
