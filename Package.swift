// swift-tools-version:5.9
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
            path: "include/stringzilla",
            sources: ["../../c/stringzilla.c"],
            publicHeadersPath: ".",
            cSettings: [
                .define("SZ_DYNAMIC_DISPATCH", to: "1"),
                .define("SZ_AVOID_LIBC", to: "0"),
                .define("SZ_DEBUG", to: "0"),
                .headerSearchPath("include/stringzilla"),
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
            sources: ["Test.swift"]
        ),
    ],
    cLanguageStandard: CLanguageStandard.c99
)
