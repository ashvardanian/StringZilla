// swift-tools-version:6.4
import PackageDescription

let package = Package(
    name: "StringZilla",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
        .watchOS(.v9),
        .visionOS(.v1),
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
            // The target is rooted at the repository so every source stays inside `path`,
            // otherwise SwiftPM silently drops entries that escape it and links nothing.
            path: ".",
            sources: [
                "c/stringzilla/runtime.c",
                "c/stringzilla/compare.c",
                "c/stringzilla/memory.c",
                "c/stringzilla/hash.c",
                "c/stringzilla/cipher.c",
                "c/stringzilla/find.c",
                "c/stringzilla/sort.c",
                "c/stringzilla/intersect.c",
                "c/stringzilla/levenshtein.c",
                "c/stringzilla/overlap.c",
                "c/stringzilla/substrings.c",
                "c/stringzilla/utf8_norm.c",
                "c/stringzilla/utf8_runes.c",
                "c/stringzilla/utf8_tokens.c",
                "c/stringzilla/utf8_wordbreaks.c",
                "c/stringzilla/utf8_graphemes.c",
                "c/stringzilla/utf8_sentences.c",
                "c/stringzilla/utf8_linebreaks.c",
                "c/stringzilla/utf8_uncased_fold.c",
                "c/stringzilla/utf8_uncased.c",
            ],
            // `include/` is the module header root, so the `module.modulemap` umbrella and the
            // `#include "stringzilla/<...>.h"` chain resolve exactly as `-I include` does in the
            // CMake, Rust, and Python builds.
            publicHeadersPath: "include",
            cSettings: [
                .define("STRINGZILLA_RUNTIME_DISPATCH", to: "1"),
                .define("STRINGZILLA_WITH_LIBC", to: "1"),
                .define("STRINGZILLA_DEBUG", to: "0"),
                .unsafeFlags(["-Wall"]),
            ]
        ),
        .target(
            name: "StringZilla",
            dependencies: ["StringZillaC"],
            path: "swift",
            exclude: ["Test.swift", "README.md"],
            sources: ["StringProtocol+StringZilla.swift"]
        ),
        .testTarget(
            name: "StringZillaTests",
            dependencies: ["StringZilla"],
            path: "swift",
            exclude: ["StringProtocol+StringZilla.swift", "README.md"],
            sources: ["Test.swift"]
        ),
    ],
    cLanguageStandard: CLanguageStandard.c99
)
