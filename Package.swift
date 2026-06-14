// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "StringZilla",
    platforms: [
        // Linux doesn't have to be explicitly listed
        .iOS(.v13),  // For iOS, version 13 and later
        .tvOS(.v13),  // For tvOS, version 13 and later
        .macOS(.v10_15),  // For macOS, version 10.15 (Catalina) and later
        .watchOS(.v6),  // For watchOS, version 6 and later
        .visionOS(.v1),  // For visionOS, version 1.0 and later
    ],
    products: [
        .library(
            name: "StringZilla",
            targets: ["StringZillaC", "StringZilla"]
        )
    ],
    targets: [
        .target(
            name: "StringZillaC",
            path: "include",
            sources: [
                "../c/stringzilla/runtime.c",
                "../c/stringzilla/compare.c",
                "../c/stringzilla/memory.c",
                "../c/stringzilla/hash.c",
                "../c/stringzilla/find.c",
                "../c/stringzilla/sort.c",
                "../c/stringzilla/intersect.c",
                "../c/stringzilla/utf8_iterate.c",
                "../c/stringzilla/utf8_case_fold.c",
                "../c/stringzilla/utf8_case_insensitive.c",
            ],
            // `include/` is the module header root, so the `module.modulemap` umbrella and the
            // `#include "stringzilla/<...>.h"` chain resolve exactly as in the CMake/Rust/Python builds (`-I include`).
            publicHeadersPath: ".",
            cSettings: [
                .define("SZ_DYNAMIC_DISPATCH", to: "1"),
                .define("SZ_AVOID_LIBC", to: "0"),
                .define("SZ_DEBUG", to: "0"),
                .unsafeFlags(["-Wall"]),
            ]
        ),
        .target(
            name: "StringZilla",
            dependencies: ["StringZillaC"],
            path: "swift",
            exclude: ["Test.swift"],
            sources: ["StringProtocol+StringZilla.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZilla"],
            path: "swift",
            exclude: ["StringProtocol+StringZilla.swift"],
            sources: ["Test.swift"]
        ),
    ],
    cLanguageStandard: CLanguageStandard.c99
)
